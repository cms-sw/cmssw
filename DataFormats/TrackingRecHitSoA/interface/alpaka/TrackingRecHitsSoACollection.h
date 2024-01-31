#ifndef DataFormats_RecHits_interface_alpakaTrackingRecHitsSoACollection
#define DataFormats_RecHits_interface_alpakaTrackingRecHitsSoACollection

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  using TrackingRecHitsSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                          TrackingRecHitHost<TrackerTraits>,
                                                          TrackingRecHitDevice<TrackerTraits, Device>>;

  //Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
  using TrackingRecHitSoAPhase1 = TrackingRecHitsSoACollection<pixelTopology::Phase1>;
  using TrackingRecHitSoAPhase2 = TrackingRecHitsSoACollection<pixelTopology::Phase2>;
  using TrackingRecHitSoAHIonPhase1 = TrackingRecHitsSoACollection<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits, typename TDevice>
  struct CopyToHost<TrackingRecHitDevice<TrackerTraits, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TrackingRecHitDevice<TrackerTraits, TDevice> const& deviceData) {
      TrackingRecHitHost<TrackerTraits> hostData(deviceData.view().metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("TrackingRecHitsSoACollection: I'm copying to host.\n");
#endif
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackingRecHitSoAPhase1, TrackingRecHitHostPhase1);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackingRecHitSoAPhase2, TrackingRecHitHostPhase2);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackingRecHitSoAHIonPhase1, TrackingRecHitHostHIonPhase1);

#endif  // DataFormats_RecHits_interface_alpakaTrackingRecHitsSoACollection