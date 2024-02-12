#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Kernel running a loop over threads/elements
// One function with multiple flavors

// The type of uniform_elements
enum class RangeType { Default, ExtentLimited, ExtentLimitedWithShift };

// The concurrency scope between threads
enum class LoopScope { Block, Grid };

// Utility for one time initializations
template <LoopScope loopScope, typename TAcc>
bool constexpr firstInLoopRange(TAcc const& acc) {
  if constexpr (loopScope == LoopScope::Block)
    return !alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
  if constexpr (loopScope == LoopScope::Grid)
    return !alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
  assert(false);
}

template <RangeType rangeType, LoopScope loopScope, typename TAcc>
size_t constexpr expectedCount(TAcc const& acc, size_t skip, size_t size) {
  if constexpr (rangeType == RangeType::ExtentLimitedWithShift)
    return skip < size ? size - skip : 0;
  else if constexpr (rangeType == RangeType::ExtentLimited)
    return size;
  else /* rangeType == RangeType::Default */
    if constexpr (loopScope == LoopScope::Block)
      return alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
    else
      return alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u];
}

template <RangeType rangeType, LoopScope loopScope>
size_t constexpr expectedCount(WorkDiv1D const& workDiv, size_t skip, size_t size) {
  if constexpr (rangeType == RangeType::ExtentLimitedWithShift)
    return skip < size ? size - skip : 0;
  else if constexpr (rangeType == RangeType::ExtentLimited)
    return size;
  else /* rangeType == RangeType::Default */
    if constexpr (loopScope == LoopScope::Block)
      return workDiv.m_blockThreadExtent[0u] * workDiv.m_threadElemExtent[0u];
    else
      return workDiv.m_gridBlockExtent[0u] * workDiv.m_blockThreadExtent[0u] * workDiv.m_threadElemExtent[0u];
}

template <RangeType rangeType, LoopScope loopScope>
struct testWordDivisionDefaultRange {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, size_t size, size_t skip, size_t* globalCounter) const {
    size_t& counter =
        (loopScope == LoopScope::Grid ? *globalCounter : alpaka::declareSharedVar<size_t, __COUNTER__>(acc));
    // Init the counter for block range. Grid range does so my mean of memset.
    if constexpr (loopScope == LoopScope::Block) {
      if (firstInLoopRange<loopScope>(acc))
        counter = 0;
      alpaka::syncBlockThreads(acc);
    }
    // The loop we are testing
    if constexpr (rangeType == RangeType::Default)
      for ([[maybe_unused]] auto idx : uniform_elements(acc))
        alpaka::atomicAdd(acc, &counter, 1ul, alpaka::hierarchy::Blocks{});
    else if constexpr (rangeType == RangeType::ExtentLimited)
      for ([[maybe_unused]] auto idx : uniform_elements(acc, size))
        alpaka::atomicAdd(acc, &counter, 1ul, alpaka::hierarchy::Blocks{});
    else if constexpr (rangeType == RangeType::ExtentLimitedWithShift)
      for ([[maybe_unused]] auto idx : uniform_elements(acc, skip, size))
        alpaka::atomicAdd(acc, &counter, 1ul, alpaka::hierarchy::Blocks{});
    alpaka::syncBlockThreads(acc);
    // Check the result. Grid range will check by memcpy-ing the result.
    if constexpr (loopScope == LoopScope::Block) {
      if (firstInLoopRange<loopScope>(acc)) {
        auto expected = expectedCount<rangeType, loopScope>(acc, skip, size);
        assert(counter == expected);
      }
    }
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

  for (auto const& device : devices) {
    // Get global memory
    Queue queue(device);
    auto counter_d = cms::alpakatools::make_device_buffer<size_t>(queue);
    auto counter_h = cms::alpakatools::make_host_buffer<size_t>(queue);
    alpaka::memset(queue, counter_d, 0);
    ssize_t BlockSize = 512;
    size_t GridSize = 4;
    for (size_t blocks = 1; blocks < GridSize * 3; blocks++)
      for (auto sizeFuzz :
           std::initializer_list<ssize_t>{-10 * BlockSize / 13, -BlockSize / 2, -1, 0, 1, BlockSize / 2})
        for (auto skip : std::initializer_list<ssize_t>{0,
                                                        1,
                                                        BlockSize / 2,
                                                        BlockSize - 1,
                                                        BlockSize,
                                                        BlockSize + 1,
                                                        BlockSize + BlockSize / 2,
                                                        2 * BlockSize - 1,
                                                        2 * BlockSize,
                                                        2 * BlockSize + 1}) {
          // Grid level iteration: we need to initialize/check at the grid level
          // Default range
          alpaka::memset(queue, counter_d, 0);
          auto workdiv = make_workdiv<Acc1D>(BlockSize, GridSize);
          alpaka::enqueue(
              queue,
              alpaka::createTaskKernel<Acc1D>(workdiv,
                                              testWordDivisionDefaultRange<RangeType::Default, LoopScope::Grid>{},
                                              blocks * BlockSize + sizeFuzz,
                                              skip,
                                              counter_d.data()));
          alpaka::memcpy(queue, counter_h, counter_d);
          alpaka::wait(queue);
          auto expected =
              expectedCount<RangeType::Default, LoopScope::Grid>(workdiv, skip, blocks * BlockSize + sizeFuzz);
          assert(*counter_h.data() == expected);

          // ExtentLimited range
          alpaka::memset(queue, counter_d, 0);
          alpaka::enqueue(
              queue,
              alpaka::createTaskKernel<Acc1D>(workdiv,
                                              testWordDivisionDefaultRange<RangeType::ExtentLimited, LoopScope::Grid>{},
                                              blocks * BlockSize + sizeFuzz,
                                              skip,
                                              counter_d.data()));
          alpaka::memcpy(queue, counter_h, counter_d);
          alpaka::wait(queue);
          expected =
              expectedCount<RangeType::ExtentLimited, LoopScope::Grid>(workdiv, skip, blocks * BlockSize + sizeFuzz);
          assert(*counter_h.data() == expected);

          // ExtentLimitedWithShift range
          alpaka::memset(queue, counter_d, 0);
          alpaka::enqueue(queue,
                          alpaka::createTaskKernel<Acc1D>(
                              workdiv,
                              testWordDivisionDefaultRange<RangeType::ExtentLimitedWithShift, LoopScope::Grid>{},
                              blocks * BlockSize + sizeFuzz,
                              skip,
                              counter_d.data()));
          alpaka::memcpy(queue, counter_h, counter_d);
          alpaka::wait(queue);
          expected = expectedCount<RangeType::ExtentLimitedWithShift, LoopScope::Grid>(
              workdiv, skip, blocks * BlockSize + sizeFuzz);
          assert(*counter_h.data() == expected);

          // Block level auto tests
          alpaka::enqueue(
              queue,
              alpaka::createTaskKernel<Acc1D>(workdiv,
                                              testWordDivisionDefaultRange<RangeType::Default, LoopScope::Grid>{},
                                              blocks * BlockSize + sizeFuzz,
                                              skip,
                                              counter_d.data()));
          alpaka::enqueue(
              queue,
              alpaka::createTaskKernel<Acc1D>(workdiv,
                                              testWordDivisionDefaultRange<RangeType::ExtentLimited, LoopScope::Grid>{},
                                              blocks * BlockSize + sizeFuzz,
                                              skip,
                                              counter_d.data()));
          alpaka::enqueue(queue,
                          alpaka::createTaskKernel<Acc1D>(
                              workdiv,
                              testWordDivisionDefaultRange<RangeType::ExtentLimitedWithShift, LoopScope::Grid>{},
                              blocks * BlockSize + sizeFuzz,
                              skip,
                              counter_d.data()));
        }
    alpaka::wait(queue);
  }
}
