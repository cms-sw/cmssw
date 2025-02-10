#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
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
  static const constexpr char* failed_msg = "failed(int) size=%d, i=%d, blockDimension=%d: c[i]=%d c[i-1]=%d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char* failed_msg = "failed(float size=%d, i=%d, blockDimension=%d: c[i]=%f c[i-1]=%f\n";
};

template <typename T>
struct testPrefixScan {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, unsigned int size) const {
    // alpaka::warp::getSize(acc) is runtime, but we need a compile-time or constexpr value
#if defined(__CUDA_ARCH__)
    // CUDA always has a warp size of 32
    auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
#elif defined(__HIP_DEVICE_COMPILE__)
    // HIP/ROCm defines warpSize as a constant expression with value 32 or 64 depending on the target device
    auto& ws = alpaka::declareSharedVar<T[warpSize], __COUNTER__>(acc);
#else
    // CPU back-ends always have a warp size of 1
    auto& ws = alpaka::declareSharedVar<T[1], __COUNTER__>(acc);
#endif
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    for (auto i : uniform_elements(acc, size)) {
      c[i] = 1;
    };

    alpaka::syncBlockThreads(acc);

    blockPrefixScan(acc, c, co, size, ws);
    blockPrefixScan(acc, c, size, ws);

    ALPAKA_ASSERT_ACC(1 == c[0]);
    ALPAKA_ASSERT_ACC(1 == co[0]);

    // TODO: not needed? Not in multi kernel version, not in CUDA version
    alpaka::syncBlockThreads(acc);

    for (auto i : uniform_elements(acc, size)) {
      if (0 == i)
        continue;
      if constexpr (!std::is_floating_point_v<T>) {
        if (!((c[i] == c[i - 1] + 1) && (c[i] == i + 1) && (c[i] == co[i])))
          printf("c[%d]=%d, co[%d]=%d\n", i, c[i], i, co[i]);
      } else {
        if (!((c[i] == c[i - 1] + 1) && (c[i] == i + 1) && (c[i] == co[i])))
          printf("c[%d]=%f, co[%d]=%f\n", i, c[i], i, co[i]);
      }
      ALPAKA_ASSERT_ACC(c[i] == c[i - 1] + 1);
      ALPAKA_ASSERT_ACC(c[i] == i + 1);
      ALPAKA_ASSERT_ACC(c[i] == co[i]);
    }
  }
};

/*
 * NB: GPU-only, so do not care about elements here.
 */
template <typename T>
struct testWarpPrefixScan {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t size) const {
    if constexpr (not requires_single_thread_per_block_v<Acc1D>) {
      ALPAKA_ASSERT_ACC(size <= static_cast<uint32_t>(alpaka::warp::getSize(acc)));
      auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
      auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

      uint32_t const blockDimension = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      uint32_t const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      auto i = blockThreadIdx;
      c[i] = 1;
      alpaka::syncBlockThreads(acc);
      // a compile-time constant would be faster, but this is more portable
      auto laneId = blockThreadIdx % alpaka::warp::getSize(acc);

      warpPrefixScan(acc, laneId, c, co, i);
      warpPrefixScan(acc, laneId, c, i);

      alpaka::syncBlockThreads(acc);

      ALPAKA_ASSERT_ACC(1 == c[0]);
      ALPAKA_ASSERT_ACC(1 == co[0]);
      if (i != 0) {
        if (c[i] != c[i - 1] + 1)
          printf(format_traits<T>::failed_msg, size, i, blockDimension, c[i], c[i - 1]);
        ALPAKA_ASSERT_ACC(c[i] == c[i - 1] + 1);
        ALPAKA_ASSERT_ACC(c[i] == static_cast<T>(i + 1));
        ALPAKA_ASSERT_ACC(c[i] == co[i]);
      }
    } else {
      // This should never be called outsie for the serial CPU backend.
      ALPAKA_ASSERT_ACC(false);
    }
  }
};

struct init {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t* v, uint32_t val, uint32_t n) const {
    for (auto index : uniform_elements(acc, n)) {
      v[index] = val;

      if (index == 0)
        printf("init\n");
    }
  }
};

struct verify {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t const* v, uint32_t n) const {
    for (auto index : uniform_elements(acc, n)) {
      ALPAKA_ASSERT_ACC(v[index] == index + 1);

      if (index == 0)
        printf("verify\n");
    }
  }
};

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();

  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  for (auto const& device : devices) {
    std::cout << "Test prefix scan on " << alpaka::getName(device) << '\n';
    auto queue = Queue(device);
    const auto warpSize = alpaka::getPreferredWarpSize(device);
    // WARP PREFIXSCAN (OBVIOUSLY GPU-ONLY)
    if constexpr (!requires_single_thread_per_block_v<Acc1D>) {
      std::cout << "warp level" << std::endl;

      const auto threadsPerBlockOrElementsPerThread = warpSize;
      const auto blocksPerGrid = 1;
      const auto workDivWarp = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

      if (warpSize >= 64)
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 64));
      if (warpSize >= 32)
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 32));
      if (warpSize >= 16)
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 12));
      if (warpSize >= 8)
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 5));
    }

    // PORTABLE BLOCK PREFIXSCAN
    std::cout << "block level" << std::endl;

    // Running kernel with 1 block, and bs threads per block or elements per thread.
    // NB: obviously for tests only, for perf would need to use bs = 1024 in GPU version.
    for (int bs = warpSize; bs <= 1024; bs += warpSize) {
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
    uint32_t num_items = 200;
    for (int ksize = 1; ksize < 4; ++ksize) {
      std::cout << "multiblock" << std::endl;
      num_items *= 10;

      auto input_d = make_device_buffer<uint32_t[]>(queue, num_items);
      auto output1_d = make_device_buffer<uint32_t[]>(queue, num_items);
      auto blockCounter_d = make_device_buffer<int32_t>(queue);

      const auto nThreadsInit = 256;  // NB: 1024 would be better
      const auto nBlocksInit = divide_up_by(num_items, nThreadsInit);
      const auto workDivMultiBlockInit = make_workdiv<Acc1D>(nBlocksInit, nThreadsInit);

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivMultiBlockInit, init(), input_d.data(), 1, num_items));
      alpaka::memset(queue, blockCounter_d, 0);

      const auto nThreads = 1024;
      const auto nBlocks = divide_up_by(num_items, nThreads);
      const auto workDivMultiBlock = make_workdiv<Acc1D>(nBlocks, nThreads);

      std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nBlocks << std::endl;
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivMultiBlock,
                                                      multiBlockPrefixScan<uint32_t>(),
                                                      input_d.data(),
                                                      output1_d.data(),
                                                      num_items,
                                                      nBlocks,
                                                      blockCounter_d.data(),
                                                      warpSize));
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivMultiBlock, verify(), output1_d.data(), num_items));

      alpaka::wait(queue);  // input_d and output1_d end of scope
    }  // ksize
  }

  return 0;
}
