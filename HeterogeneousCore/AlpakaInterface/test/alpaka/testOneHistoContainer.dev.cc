#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <int NBINS, int S, int DELTA>
struct mykernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, T const* __restrict__ v, uint32_t N) const {
    ALPAKA_ASSERT_ACC(v);
    ALPAKA_ASSERT_ACC(N == 12000);

    const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    if (threadIdxLocal == 0) {
      printf("start kernel for %d data\n", N);
    }

    using Hist = HistoContainer<T, NBINS, 12000, S, uint16_t>;

    auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
    auto& ws = alpaka::declareSharedVar<typename Hist::Counter[32], __COUNTER__>(acc);

    // set off zero
    for (auto j : uniform_elements(acc, Hist::totbins())) {
      hist.off[j] = 0;
    }
    alpaka::syncBlockThreads(acc);

    // set bins zero
    for (auto j : uniform_elements(acc, Hist::totbins())) {
      hist.content[j] = 0;
    }
    alpaka::syncBlockThreads(acc);

    // count
    for (auto j : uniform_elements(acc, N)) {
      hist.count(acc, v[j]);
    }
    alpaka::syncBlockThreads(acc);

    ALPAKA_ASSERT_ACC(0 == hist.size());
    alpaka::syncBlockThreads(acc);

    // finalize
    hist.finalize(acc, ws);
    alpaka::syncBlockThreads(acc);

    ALPAKA_ASSERT_ACC(N == hist.size());

    // verify
    for ([[maybe_unused]] auto j : uniform_elements(acc, Hist::nbins())) {
      ALPAKA_ASSERT_ACC(hist.off[j] <= hist.off[j + 1]);
    }
    alpaka::syncBlockThreads(acc);

    for (auto j : uniform_elements(acc, 32)) {
      ws[j] = 0;  // used by prefix scan...
    }
    alpaka::syncBlockThreads(acc);

    // fill
    for (auto j : uniform_elements(acc, N)) {
      hist.fill(acc, v[j], j);
    }
    alpaka::syncBlockThreads(acc);

    ALPAKA_ASSERT_ACC(0 == hist.off[0]);
    ALPAKA_ASSERT_ACC(N == hist.size());

    // bin
#ifndef NDEBUG
    for (auto j : uniform_elements(acc, hist.size() - 1)) {
      auto p = hist.begin() + j;
      ALPAKA_ASSERT_ACC((*p) < N);
      [[maybe_unused]] auto k1 = Hist::bin(v[*p]);
      [[maybe_unused]] auto k2 = Hist::bin(v[*(p + 1)]);
      ALPAKA_ASSERT_ACC(k2 >= k1);
    }
#endif

    // forEachInWindow
    for (auto i : uniform_elements(acc, hist.size())) {
      auto p = hist.begin() + i;
      auto j = *p;
#ifndef NDEBUG
      auto b0 = Hist::bin(v[j]);
#endif
      [[maybe_unused]] int tot = 0;
      auto ftest = [&](unsigned int k) {
        ALPAKA_ASSERT_ACC(k < N);
        ++tot;
      };
      forEachInWindow(hist, v[j], v[j], ftest);
#ifndef NDEBUG
      [[maybe_unused]] int rtot = hist.size(b0);
      ALPAKA_ASSERT_ACC(tot == rtot);
#endif
      tot = 0;
      auto vm = int(v[j]) - DELTA;
      auto vp = int(v[j]) + DELTA;
      constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
      vm = std::max(vm, 0);
      vm = std::min(vm, vmax);
      vp = std::min(vp, vmax);
      vp = std::max(vp, 0);
      ALPAKA_ASSERT_ACC(vp >= vm);
      forEachInWindow(hist, vm, vp, ftest);
#ifndef NDEBUG
      int bp = Hist::bin(vp);
      int bm = Hist::bin(vm);
      rtot = hist.end(bp) - hist.begin(bm);
      ALPAKA_ASSERT_ACC(tot == rtot);
#endif
    }
  }
};

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go(const DevHost& host, const Device& device, Queue& queue) {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);
  constexpr unsigned int N = 12000;

  using Hist = HistoContainer<T, NBINS, N, S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist{}.capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  auto v = make_host_buffer<T[]>(queue, N);
  auto v_d = make_device_buffer<T[]>(queue, N);

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;

    alpaka::memcpy(queue, v_d, v);

    const auto threadsPerBlockOrElementsPerThread = 256u;
    const auto blocksPerGrid = 1u;
    const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, mykernel<NBINS, S, DELTA>(), v_d.data(), N));
  }
  alpaka::wait(queue);
}

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << ", the test will be skipped.\n";
    return 0;
  }

  auto const& host = cms::alpakatools::host();

  // run the test on each device
  for (auto const& device : devices) {
    std::cout << "Test One Histo Container on " << alpaka::getName(device) << '\n';

    auto queue = Queue(device);

    go<int16_t>(host, device, queue);
    go<uint8_t, 128, 8, 4>(host, device, queue);
    go<uint16_t, 313 / 2, 9, 4>(host, device, queue);
  }

  return 0;
}
