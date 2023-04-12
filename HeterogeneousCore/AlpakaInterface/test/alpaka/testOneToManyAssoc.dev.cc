#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using Assoc = OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = OneToManyAssoc<uint16_t, 128, MaxAssocs>;
using Multiplicity = OneToManyAssoc<uint16_t, 8, MaxTk>;
using TK = std::array<uint16_t, 4>;

struct countMultiLocal {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t i) {
      auto& local = alpaka::declareSharedVar<Multiplicity::CountersOnly, __COUNTER__>(acc);
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const bool oncePerSharedMemoryAccess = (threadIdxLocal == 0);
      if (oncePerSharedMemoryAccess) {
        local.zero();
      }
      alpaka::syncBlockThreads(acc);
      local.countDirect(acc, 2 + i % 4);
      alpaka::syncBlockThreads(acc);
      if (oncePerSharedMemoryAccess) {
        assoc->add(acc, local);
      }
    });
  }
};

struct countMulti {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t i) { assoc->countDirect(acc, 2 + i % 4); });
  }
};

struct verifyMulti {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2) const {
    for_each_element_in_grid_strided(
        acc, Multiplicity::totbins(), [&](uint32_t i) { assert(m1->off[i] == m2->off[i]); });
  }
};

struct count {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, 4 * n, [&](uint32_t i) {
      auto k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->countDirect(acc, tk[k][j]);
      }
    });
  }
};

struct fill {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, 4 * n, [&](uint32_t i) {
      auto k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->fillDirect(acc, tk[k][j], k);
      }
    });
  }
};

struct verify {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Assoc* __restrict__ assoc) const {
    assert(assoc->size() < Assoc::capacity());
  }
};

struct fillBulk {
  template <typename TAcc, typename Assoc>
  ALPAKA_FN_ACC void operator()(
      const TAcc& acc, AtomicPairCounter* apc, TK const* __restrict__ tk, Assoc* __restrict__ assoc, uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t k) {
      auto m = tk[k][3] < MaxElem ? 4 : 3;
      assoc->bulkFill(acc, *apc, &tk[k][0], m);
    });
  }
};

struct verifyBulk {
  template <typename TAcc, typename Assoc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Assoc const* __restrict__ assoc, AtomicPairCounter const* apc) const {
    if (apc->get().m >= Assoc::nbins()) {
      printf("Overflow %d %d\n", apc->get().m, Assoc::nbins());
    }
    assert(assoc->size() < Assoc::capacity());
  }
};

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << ", the test will be skipped.\n";
    return 0;
  }

  // run the test on each device
  for (auto const& device : devices) {
    Queue queue(device);

    std::cout << "OneToManyAssoc " << sizeof(Assoc) << ' ' << Assoc::nbins() << ' ' << Assoc::capacity() << std::endl;
    std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << ' ' << SmallAssoc::nbins() << ' '
              << SmallAssoc::capacity() << std::endl;

    std::mt19937 eng;
    std::geometric_distribution<int> rdm(0.8);

    constexpr uint32_t N = 4000;

    auto tr = make_host_buffer<std::array<uint16_t, 4>[]>(queue, N);
    // fill with "index" to element
    long long ave = 0;
    int imax = 0;
    auto n = 0U;
    auto z = 0U;
    auto nz = 0U;
    for (auto i = 0U; i < 4U; ++i) {
      auto j = 0U;
      while (j < N && n < MaxElem) {
        if (z == 11) {
          ++n;
          z = 0;
          ++nz;
          continue;
        }  // a bit of not assoc
        auto x = rdm(eng);
        auto k = std::min(j + x + 1, N);
        if (i == 3 && z == 3) {  // some triplets time to time
          for (; j < k; ++j)
            tr[j][i] = MaxElem + 1;
        } else {
          ave += x + 1;
          imax = std::max(imax, x);
          for (; j < k; ++j)
            tr[j][i] = n;
          ++n;
        }
        ++z;
      }
      assert(n <= MaxElem);
      assert(j <= N);
    }
    std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

    auto v_d = make_device_buffer<std::array<uint16_t, 4>[]>(queue, N);
    alpaka::memcpy(queue, v_d, tr);

    auto a_d = make_device_buffer<Assoc>(queue);
    alpaka::memset(queue, a_d, 0);

    const auto threadsPerBlockOrElementsPerThread = 256u;
    const auto blocksPerGrid4N = divide_up_by(4 * N, threadsPerBlockOrElementsPerThread);
    const auto workDiv4N = make_workdiv<Acc1D>(blocksPerGrid4N, threadsPerBlockOrElementsPerThread);

    launchZero<Acc1D>(a_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, count(), v_d.data(), a_d.data(), N));

    launchFinalize<Acc1D>(a_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verify(), a_d.data()));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, fill(), v_d.data(), a_d.data(), N));

    auto la = make_host_buffer<Assoc>(queue);
    alpaka::memcpy(queue, la, a_d);
    alpaka::wait(queue);

    std::cout << la->size() << std::endl;
    imax = 0;
    ave = 0;
    z = 0;
    for (auto i = 0U; i < n; ++i) {
      auto x = la->size(i);
      if (x == 0) {
        z++;
        continue;
      }
      ave += x;
      imax = std::max(imax, int(x));
    }
    assert(0 == la->size(n));
    std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

    // now the inverse map (actually this is the direct....)
    auto dc_d = make_device_buffer<AtomicPairCounter>(queue);
    alpaka::memset(queue, dc_d, 0);

    const auto blocksPerGrid = divide_up_by(N, threadsPerBlockOrElementsPerThread);
    const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv, fillBulk(), dc_d.data(), v_d.data(), a_d.data(), N));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, finalizeBulk(), dc_d.data(), a_d.data()));

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verifyBulk(), a_d.data(), dc_d.data()));

    alpaka::memcpy(queue, la, a_d);

    auto dc = make_host_buffer<AtomicPairCounter>(queue);
    alpaka::memcpy(queue, dc, dc_d);
    alpaka::wait(queue);

    alpaka::memset(queue, dc_d, 0);
    auto sa_d = make_device_buffer<SmallAssoc>(queue);
    alpaka::memset(queue, sa_d, 0);

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv, fillBulk(), dc_d.data(), v_d.data(), sa_d.data(), N));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, finalizeBulk(), dc_d.data(), sa_d.data()));

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verifyBulk(), sa_d.data(), dc_d.data()));

    std::cout << "final counter value " << dc->get().n << ' ' << dc->get().m << std::endl;

    std::cout << la->size() << std::endl;
    imax = 0;
    ave = 0;
    for (auto i = 0U; i < N; ++i) {
      auto x = la->size(i);
      if (!(x == 4 || x == 3)) {
        std::cout << i << ' ' << x << std::endl;
      }
      assert(x == 4 || x == 3);
      ave += x;
      imax = std::max(imax, int(x));
    }
    assert(0 == la->size(N));
    std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

    // here verify use of block local counters
    auto m1_d = make_device_buffer<Multiplicity>(queue);
    alpaka::memset(queue, m1_d, 0);
    auto m2_d = make_device_buffer<Multiplicity>(queue);
    alpaka::memset(queue, m2_d, 0);

    launchZero<Acc1D>(m1_d.data(), queue);
    launchZero<Acc1D>(m2_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, countMulti(), v_d.data(), m1_d.data(), N));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, countMultiLocal(), v_d.data(), m2_d.data(), N));

    const auto blocksPerGridTotBins = 1u;
    const auto threadsPerBlockOrElementsPerThreadTotBins = Multiplicity::totbins();
    const auto workDivTotBins = make_workdiv<Acc1D>(blocksPerGridTotBins, threadsPerBlockOrElementsPerThreadTotBins);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivTotBins, verifyMulti(), m1_d.data(), m2_d.data()));

    launchFinalize<Acc1D>(m1_d.data(), queue);
    launchFinalize<Acc1D>(m2_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivTotBins, verifyMulti(), m1_d.data(), m2_d.data()));

    alpaka::wait(queue);

    return 0;
  }
}