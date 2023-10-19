#ifndef DataFormats_RecHits_interface_alpaka_TrackingRecHitSoACollection_h
#define DataFormats_RecHits_interface_alpaka_TrackingRecHitSoACollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoADevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename TrackerTraits>
  using TrackingRecHitAlpakaCollection = TrackingRecHitHost<TrackerTraits>;
#else
  template <typename TrackerTraits>
  using TrackingRecHitAlpakaCollection = TrackingRecHitDevice<TrackerTraits, Device>;
#endif
  //Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
  using TrackingRecHitAlpakaSoAPhase1 = TrackingRecHitAlpakaCollection<pixelTopology::Phase1>;
  using TrackingRecHitAlpakaSoAPhase2 = TrackingRecHitAlpakaCollection<pixelTopology::Phase2>;
  using TrackingRecHitAlpakaSoAHIonPhase1 = TrackingRecHitAlpakaCollection<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitAlpakaCollection<TrackerTraits>> {
    template <typename TQueue>
    static auto copyAsync(
        TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitAlpakaCollection<TrackerTraits> const& deviceData) {
      TrackingRecHitHost<TrackerTraits> hostData(deviceData.view().metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_RecHits_interface_alpaka_TrackingRecHitSoACollection_h