#ifndef DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsCollection_h
#define DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsCollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelDigiErrorsCollection = SiPixelDigiErrorsHost;
#else
  using SiPixelDigiErrorsCollection = SiPixelDigiErrorsDevice<Device>;
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsCollection> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsCollection const& srcData) {
      SiPixelDigiErrorsHost dstData(srcData.maxFedWords(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());

      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsCollection_h
