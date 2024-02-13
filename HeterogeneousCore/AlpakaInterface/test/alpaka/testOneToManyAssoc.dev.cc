#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using AssocRandomAccess = OneToManyAssocRandomAccess<uint16_t, MaxElem, MaxAssocs>;
using AssocSequential = OneToManyAssocSequential<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = OneToManyAssocSequential<uint16_t, 128, MaxAssocs>;
using Multiplicity = OneToManyAssocRandomAccess<uint16_t, 8, MaxTk>;
using TK = std::array<uint16_t, 4>;

namespace {
  template <typename T>
  ALPAKA_FN_HOST_ACC typename std::make_signed<T>::type toSigned(T v) {
    return static_cast<typename std::make_signed<T>::type>(v);
  }
}  // namespace

struct countMultiLocal {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    for (auto i : uniform_elements(acc, n)) {
      auto& local = alpaka::declareSharedVar<Multiplicity::CountersOnly, __COUNTER__>(acc);
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const bool oncePerSharedMemoryAccess = (threadIdxLocal == 0);
      if (oncePerSharedMemoryAccess) {
        local.zero();
      }
      alpaka::syncBlockThreads(acc);
      local.count(acc, 2 + i % 4);
      alpaka::syncBlockThreads(acc);
      if (oncePerSharedMemoryAccess) {
        assoc->add(acc, local);
      }
    }
  }
};

struct countMulti {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    for (auto i : uniform_elements(acc, n)) {
      assoc->count(acc, 2 + i % 4);
    }
  }
};

struct verifyMulti {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2) const {
    for ([[maybe_unused]] auto i : uniform_elements(acc, Multiplicity{}.totOnes())) {
      ALPAKA_ASSERT_ACC(m1->off[i] == m2->off[i]);
    }
  }
};

struct count {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                AssocRandomAccess* __restrict__ assoc,
                                uint32_t n) const {
    for (auto i : uniform_elements(acc, 4 * n)) {
      auto k = i / 4;
      auto j = i - 4 * k;
      ALPAKA_ASSERT_ACC(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->count(acc, tk[k][j]);
      }
    }
  }
};

struct fill {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                AssocRandomAccess* __restrict__ assoc,
                                uint32_t n) const {
    for (auto i : uniform_elements(acc, 4 * n)) {
      auto k = i / 4;
      auto j = i - 4 * k;
      ALPAKA_ASSERT_ACC(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->fill(acc, tk[k][j], k);
      }
    }
  }
};

struct verify {
  template <typename TAcc, typename Assoc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Assoc* __restrict__ assoc) const {
    ALPAKA_ASSERT_ACC(assoc->size() < Assoc{}.capacity());
  }
};

struct fillBulk {
  template <typename TAcc, typename Assoc>
  ALPAKA_FN_ACC void operator()(
      const TAcc& acc, AtomicPairCounter* apc, TK const* __restrict__ tk, Assoc* __restrict__ assoc, uint32_t n) const {
    for (auto k : uniform_elements(acc, n)) {
      auto m = tk[k][3] < MaxElem ? 4 : 3;
      assoc->bulkFill(acc, *apc, &tk[k][0], m);
    }
  }
};

struct verifyBulk {
  template <typename TAcc, typename Assoc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Assoc const* __restrict__ assoc, AtomicPairCounter const* apc) const {
    if (::toSigned(apc->get().first) >= Assoc::ctNOnes()) {
      printf("Overflow %d %d\n", apc->get().first, Assoc::ctNOnes());
    }
    ALPAKA_ASSERT_ACC(toSigned(assoc->size()) < Assoc::ctCapacity());
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

    std::cout << "OneToManyAssocRandomAccess " << sizeof(AssocRandomAccess) << " Ones=" << AssocRandomAccess{}.totOnes()
              << " Capacity=" << AssocRandomAccess{}.capacity() << std::endl;
    std::cout << "OneToManyAssocSequential " << sizeof(AssocSequential) << " Ones=" << AssocSequential{}.totOnes()
              << " Capacity=" << AssocSequential{}.capacity() << std::endl;
    std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << " Ones=" << SmallAssoc{}.totOnes()
              << " Capacity=" << SmallAssoc{}.capacity() << std::endl;

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
      ALPAKA_ASSERT_ACC(n <= MaxElem);
      ALPAKA_ASSERT_ACC(j <= N);
    }
    std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

    auto v_d = make_device_buffer<std::array<uint16_t, 4>[]>(queue, N);
    alpaka::memcpy(queue, v_d, tr);

    auto ara_d = make_device_buffer<AssocRandomAccess>(queue);
    alpaka::memset(queue, ara_d, 0);

    const auto threadsPerBlockOrElementsPerThread = 256u;
    const auto blocksPerGrid4N = divide_up_by(4 * N, threadsPerBlockOrElementsPerThread);
    const auto workDiv4N = make_workdiv<Acc1D>(blocksPerGrid4N, threadsPerBlockOrElementsPerThread);

    AssocRandomAccess::template launchZero<Acc1D>(ara_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, count(), v_d.data(), ara_d.data(), N));

    AssocRandomAccess::template launchFinalize<Acc1D>(ara_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verify(), ara_d.data()));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, fill(), v_d.data(), ara_d.data(), N));

    auto ara_h = make_host_buffer<AssocRandomAccess>(queue);
    alpaka::memcpy(queue, ara_h, ara_d);
    alpaka::wait(queue);

    std::cout << ara_h->size() << std::endl;
    imax = 0;
    ave = 0;
    z = 0;
    for (auto i = 0U; i < n; ++i) {
      auto x = ara_h->size(i);
      if (x == 0) {
        z++;
        continue;
      }
      ave += x;
      imax = std::max(imax, int(x));
    }
    ALPAKA_ASSERT_ACC(0 == ara_h->size(n));
    std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

    // now the inverse map (actually this is the direct....)
    auto dc_d = make_device_buffer<AtomicPairCounter>(queue);
    alpaka::memset(queue, dc_d, 0);

    const auto blocksPerGrid = divide_up_by(N, threadsPerBlockOrElementsPerThread);
    const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

    auto as_d = make_device_buffer<AssocSequential>(queue);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv, fillBulk(), dc_d.data(), v_d.data(), as_d.data(), N));

    alpaka::enqueue(
        queue, alpaka::createTaskKernel<Acc1D>(workDiv, AssocSequential::finalizeBulk(), dc_d.data(), as_d.data()));

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verifyBulk(), as_d.data(), dc_d.data()));

    auto as_h = make_host_buffer<AssocSequential>(queue);
    alpaka::memcpy(queue, as_h, as_d);

    auto dc_h = make_host_buffer<AtomicPairCounter>(queue);
    alpaka::memcpy(queue, dc_h, dc_d);
    alpaka::wait(queue);

    alpaka::memset(queue, dc_d, 0);
    auto sa_d = make_device_buffer<SmallAssoc>(queue);
    alpaka::memset(queue, sa_d, 0);

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv, fillBulk(), dc_d.data(), v_d.data(), sa_d.data(), N));

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv, SmallAssoc::finalizeBulk(), dc_d.data(), sa_d.data()));

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verifyBulk(), sa_d.data(), dc_d.data()));

    std::cout << "final counter value " << dc_h->get().second << ' ' << dc_h->get().first << std::endl;

    std::cout << as_h->size() << std::endl;
    imax = 0;
    ave = 0;
    for (auto i = 0U; i < N; ++i) {
      auto x = as_h->size(i);
      if (!(x == 4 || x == 3)) {
        std::cout << "i=" << i << " x=" << x << std::endl;
      }
      ALPAKA_ASSERT_ACC(x == 4 || x == 3);
      ave += x;
      imax = std::max(imax, int(x));
    }
    ALPAKA_ASSERT_ACC(0 == as_h->size(N));
    std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

    // here verify use of block local counters
    auto m1_d = make_device_buffer<Multiplicity>(queue);
    alpaka::memset(queue, m1_d, 0);
    auto m2_d = make_device_buffer<Multiplicity>(queue);
    alpaka::memset(queue, m2_d, 0);

    Multiplicity::template launchZero<Acc1D>(m1_d.data(), queue);
    Multiplicity::template launchZero<Acc1D>(m2_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, countMulti(), v_d.data(), m1_d.data(), N));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, countMultiLocal(), v_d.data(), m2_d.data(), N));

    const auto blocksPerGridTotBins = 1u;
    const auto threadsPerBlockOrElementsPerThreadTotBins = Multiplicity::ctNOnes();
    const auto workDivTotBins = make_workdiv<Acc1D>(blocksPerGridTotBins, threadsPerBlockOrElementsPerThreadTotBins);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivTotBins, verifyMulti(), m1_d.data(), m2_d.data()));

    Multiplicity::launchFinalize<Acc1D>(m1_d.data(), queue);
    Multiplicity::launchFinalize<Acc1D>(m2_d.data(), queue);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivTotBins, verifyMulti(), m1_d.data(), m2_d.data()));

    alpaka::wait(queue);

    return 0;
  }
}
