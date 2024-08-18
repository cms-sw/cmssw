#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsSoACollection_h

#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  using TrackingRecHitsSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                          TrackingRecHitHost<TrackerTraits>,
                                                          TrackingRecHitDevice<TrackerTraits, Device>>;

  // Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
  using TrackingRecHitSoAPhase1 = TrackingRecHitsSoACollection<pixelTopology::Phase1>;
  using TrackingRecHitSoAPhase2 = TrackingRecHitsSoACollection<pixelTopology::Phase2>;
  using TrackingRecHitSoAHIonPhase1 = TrackingRecHitsSoACollection<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits, typename TDevice>
  struct CopyToHost<TrackingRecHitDevice<TrackerTraits, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TrackingRecHitDevice<TrackerTraits, TDevice> const& deviceData) {
      TrackingRecHitHost<TrackerTraits> hostData(queue, deviceData.view().metadata().size());
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("TrackingRecHitsSoACollection: I'm copying to host.\n");
      alpaka::wait(queue);
      assert(deviceData.nHits() == hostData.nHits());
      assert(deviceData.offsetBPIX2() == hostData.offsetBPIX2());
#endif
      // Update the contents address of the phiBinner histo container after the copy from device happened
      alpaka::wait(queue);
      typename TrackingRecHitSoA<TrackerTraits>::PhiBinnerView pbv;
      pbv.assoc = &(hostData.view().phiBinner());
      pbv.offSize = -1;
      pbv.offStorage = nullptr;
      pbv.contentSize = hostData.nHits();
      pbv.contentStorage = hostData.view().phiBinnerStorage();
      hostData.view().phiBinner().initStorage(pbv);

      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackingRecHitSoAPhase1, TrackingRecHitHostPhase1);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackingRecHitSoAPhase2, TrackingRecHitHostPhase2);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackingRecHitSoAHIonPhase1, TrackingRecHitHostHIonPhase1);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsSoACollection_h
