#ifndef DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigisCollection_h
#define DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigisCollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelDigisCollection = SiPixelDigisHost;
#else
  using SiPixelDigisCollection = SiPixelDigisDevice<Device>;
#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisCollection> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisCollection const &srcData) {
      SiPixelDigisHost dstData(srcData.view().metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNModulesDigis(srcData.nModules(), srcData.nDigis());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigisCollection_h
