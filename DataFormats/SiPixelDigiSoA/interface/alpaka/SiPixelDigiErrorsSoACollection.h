#ifndef DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsSoACollection_h
#define DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelDigiErrorsSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, SiPixelDigiErrorsHost, SiPixelDigiErrorsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<SiPixelDigiErrorsDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, SiPixelDigiErrorsDevice<TDevice> const& srcData) {
      SiPixelDigiErrorsHost dstData(srcData.maxFedWords(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
#ifdef GPU_DEBUG
      printf("SiPixelDigiErrorsSoACollection: I'm copying to host.\n");
#endif
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiPixelDigiErrorsSoACollection, SiPixelDigiErrorsHost);

#endif  // DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsSoACollection_h
