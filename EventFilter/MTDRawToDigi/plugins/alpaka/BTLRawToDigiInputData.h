#ifndef EventFilter_MTDRawToDigi_plugins_alpaka_BTLRawToDigiInputData_h
#define EventFilter_MTDRawToDigi_plugins_alpaka_BTLRawToDigiInputData_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct BTLRawToDigiInputData {
    BTLRawToDigiInputData() = delete;

    explicit BTLRawToDigiInputData(const Queue& queue, size_t channelCapacity)
        : rawWords{cms::alpakatools::make_host_buffer<uint64_t[]>(queue, 2 * channelCapacity)},
          channelFedId{cms::alpakatools::make_host_buffer<int32_t[]>(queue, channelCapacity)} {}

    cms::alpakatools::host_buffer<uint64_t[]> rawWords;
    cms::alpakatools::host_buffer<int32_t[]> channelFedId;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
