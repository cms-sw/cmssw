#ifndef DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigisSoACollection_h
#define DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigisSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelDigisSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, SiPixelDigisHost, SiPixelDigisDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<SiPixelDigisDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, SiPixelDigisDevice<TDevice> const &srcData) {
      SiPixelDigisHost dstData(srcData.view().metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNModulesDigis(srcData.nModules(), srcData.nDigis());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiPixelDigisSoACollection, SiPixelDigisHost);

#endif  // DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigisSoACollection_h
