#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// static constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(alpakaTestPrefixScan) "]";

template <typename T>
struct format_traits {
public:
  static const constexpr char* failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char* failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T>
struct testPrefixScan {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, unsigned int size) const {
    auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    for_each_element_in_block_strided(acc, size, [&](uint32_t i) { c[i] = 1; });

    alpaka::syncBlockThreads(acc);

    blockPrefixScan(acc, c, co, size, ws);
    blockPrefixScan(acc, c, size, ws);

    assert(1 == c[0]);
    assert(1 == co[0]);

    for_each_element_in_block_strided(acc, size, 1u, [&](uint32_t i) {
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == i + 1);
      assert(c[i] == co[i]);
    });
  }
};

/*
 * NB: GPU-only, so do not care about elements here.
 */
template <typename T>
struct testWarpPrefixScan {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, uint32_t size) const {
#if defined(ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND) && defined(__CUDA_ARCH__) || \
    defined(ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND) && defined(__HIP_DEVICE_COMPILE__)
    assert(size <= 32);
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    uint32_t const blockDimension = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    uint32_t const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    auto i = blockThreadIdx;
    c[i] = 1;
    alpaka::syncBlockThreads(acc);
    auto laneId = blockThreadIdx & 0x1f;

    warpPrefixScan(laneId, c, co, i, 0xffffffff);
    warpPrefixScan(laneId, c, i, 0xffffffff);

    alpaka::syncBlockThreads(acc);

    assert(1 == c[0]);
    assert(1 == co[0]);
    if (i != 0) {
      if (c[i] != c[i - 1] + 1)
        printf(format_traits<T>::failed_msg, size, i, blockDimension, c[i], c[i - 1]);
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == static_cast<T>(i + 1));
      assert(c[i] == co[i]);
    }
#endif
  }
};

struct init {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, uint32_t* v, uint32_t val, uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t index) {
      v[index] = val;

      if (index == 0)
        printf("init\n");
    });
  }
};

struct verify {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, uint32_t const* v, uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t index) {
      assert(v[index] == index + 1);

      if (index == 0)
        printf("verify\n");
    });
  }
};

int main() {

  // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    // auto const& host = cms::alpakatools::host();

    if (devices.empty()) {
      std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
                << ", the test will be skipped.\n";
      return 0;
    }

    for (auto const& device : devices) 
    {
      std::cout << "Test prefix scan on " << alpaka::getName(device) << '\n';
      auto queue = Queue(device);
      // WARP PREFIXSCAN (OBVIOUSLY GPU-ONLY)
#if defined(ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND) || defined(ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND)
  std::cout << "warp level" << std::endl;

  const auto threadsPerBlockOrElementsPerThread = 32;
  const auto blocksPerGrid = 1;
  const auto workDivWarp = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 32));
  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 16));
  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 5));
#endif

  // PORTABLE BLOCK PREFIXSCAN
  std::cout << "block level" << std::endl;

  // Running kernel with 1 block, and bs threads per block or elements per thread.
  // NB: obviously for tests only, for perf would need to use bs = 1024 in GPU version.
  for (int bs = 32; bs <= 1024; bs += 32) {
    const auto blocksPerGrid2 = 1;
    const auto workDivSingleBlock = make_workdiv<Acc1D>(blocksPerGrid2, bs);

    std::cout << "blocks per grid: " << blocksPerGrid2 << ", threads per block or elements per thread: " << bs
              << std::endl;

    // Problem size
    for (int j = 1; j <= 1024; ++j) {
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivSingleBlock, testPrefixScan<uint16_t>(), j));
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivSingleBlock, testPrefixScan<float>(), j));
    }
  }

      // PORTABLE MULTI-BLOCK PREFIXSCAN
      int num_items = 200;
      for (int ksize = 1; ksize < 4; ++ksize) {
        std::cout << "multiblock" << std::endl;
        num_items *= 10;

        auto input_d = make_device_buffer<uint32_t[]>(queue, num_items);
        auto output1_d = make_device_buffer<uint32_t[]>(queue, num_items);

        const auto nThreadsInit = 256;  // NB: 1024 would be better
        const auto nBlocksInit = divide_up_by(num_items, nThreadsInit);
        const auto workDivMultiBlockInit = make_workdiv<Acc1D>(nBlocksInit, nThreadsInit);

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivMultiBlockInit, init(), input_d.data(), 1, num_items));

        const auto nThreads = 1024;
        const auto nBlocks = divide_up_by(num_items, nThreads);
        const auto workDivMultiBlock = make_workdiv<Acc1D>(nBlocks, nThreads);

        std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nBlocks << std::endl;
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc1D>(
                workDivMultiBlock, multiBlockPrefixScanFirstStep<uint32_t>(), input_d.data(), output1_d.data(), num_items));

        const auto blocksPerGridSecondStep = 1;
        const auto workDivMultiBlockSecondStep = make_workdiv<Acc1D>(blocksPerGridSecondStep, nThreads);
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivMultiBlockSecondStep,
                                                        multiBlockPrefixScanSecondStep<uint32_t>(),
                                                        input_d.data(),
                                                        output1_d.data(),
                                                        num_items,
                                                        nBlocks));

        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivMultiBlock, verify(), output1_d.data(), num_items));

        alpaka::wait(queue);  // input_d and output1_d end of scope
      }                       // ksize

    }

  return 0;
}