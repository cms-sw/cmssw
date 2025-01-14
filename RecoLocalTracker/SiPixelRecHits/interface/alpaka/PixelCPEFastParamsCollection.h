#ifndef RecoLocalTracker_SiPixelRecHits_interface_alpaka_PixelCPEFastParamsCollection_h
#define RecoLocalTracker_SiPixelRecHits_interface_alpaka_PixelCPEFastParamsCollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsDevice.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  using PixelCPEFastParams = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                PixelCPEFastParamsHost<TrackerTraits>,
                                                PixelCPEFastParamsDevice<Device, TrackerTraits>>;

  using PixelCPEFastParamsPhase1 = PixelCPEFastParams<pixelTopology::Phase1>;
  using PixelCPEFastParamsHIonPhase1 = PixelCPEFastParams<pixelTopology::HIonPhase1>;
  using PixelCPEFastParamsPhase2 = PixelCPEFastParams<pixelTopology::Phase2>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits>
  struct CopyToDevice<PixelCPEFastParamsHost<TrackerTraits>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PixelCPEFastParamsHost<TrackerTraits> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PixelCPEFastParamsDevice<TDevice, TrackerTraits> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_PixelCPEFastParamsoA_interface_alpaka_PixelCPEFastParamsCollection_h
