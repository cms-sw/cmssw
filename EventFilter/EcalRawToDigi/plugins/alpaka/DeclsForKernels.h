#ifndef EventFilter_EcalRawToDigi_plugins_alpaka_DeclsForKernels_h
#define EventFilter_EcalRawToDigi_plugins_alpaka_DeclsForKernels_h

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::raw {

  struct InputDataHost {
    // delete the default constructor because alpaka buffers do not have a default constructor
    InputDataHost() = delete;

    explicit InputDataHost(const Queue& queue, size_t size, size_t nFeds)
        : data{cms::alpakatools::make_host_buffer<unsigned char[]>(queue, size)},
          offsets{cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nFeds)},
          feds{cms::alpakatools::make_host_buffer<int[]>(queue, nFeds)} {};

    cms::alpakatools::host_buffer<unsigned char[]> data;
    cms::alpakatools::host_buffer<uint32_t[]> offsets;
    cms::alpakatools::host_buffer<int[]> feds;
  };

  struct ConfigurationParameters {
    uint32_t maxChannelsEE;
    uint32_t maxChannelsEB;
  };

  struct InputDataDevice {
    InputDataDevice() = delete;

    explicit InputDataDevice(const Queue& queue, size_t size, size_t nFeds)
        : data{cms::alpakatools::make_device_buffer<unsigned char[]>(queue, size)},
          offsets{cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nFeds)},
          feds{cms::alpakatools::make_device_buffer<int[]>(queue, nFeds)} {};

    cms::alpakatools::device_buffer<Device, unsigned char[]> data;
    cms::alpakatools::device_buffer<Device, uint32_t[]> offsets;
    cms::alpakatools::device_buffer<Device, int[]> feds;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::raw

#endif  // EventFilter_EcalRawToDigi_plugins_alpaka_DeclsForKernels_h
