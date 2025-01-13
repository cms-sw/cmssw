#ifndef DataFormats_TrackSoA_interface_alpaka_TracksSoACollection_h
#define DataFormats_TrackSoA_interface_alpaka_TracksSoACollection_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  using TracksSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                 TracksHost<TrackerTraits>,
                                                 TracksDevice<TrackerTraits, Device>>;

  //Classes definition for Phase1/Phase2/HIonPhase1, to make the classes_def lighter. Not actually used in the code.
  namespace pixelTrack {
    using TracksSoACollectionPhase1 = TracksSoACollection<pixelTopology::Phase1>;
    using TracksSoACollectionPhase2 = TracksSoACollection<pixelTopology::Phase2>;
    using TracksSoACollectionHIonPhase1 = TracksSoACollection<pixelTopology::HIonPhase1>;
    using TracksSoACollectionPhase1Strip = TracksSoACollection<pixelTopology::Phase1Strip>;
  }  // namespace pixelTrack
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits, typename TDevice>
  struct CopyToHost<TracksDevice<TrackerTraits, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TracksDevice<TrackerTraits, TDevice> const& deviceData) {
      ::TracksHost<TrackerTraits> hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("TracksSoACollection: I'm copying to host.\n");
#endif
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(pixelTrack::TracksSoACollectionPhase1, pixelTrack::TracksHostPhase1);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(pixelTrack::TracksSoACollectionPhase2, pixelTrack::TracksHostPhase2);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(pixelTrack::TracksSoACollectionHIonPhase1, pixelTrack::TracksHostHIonPhase1);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(pixelTrack::TracksSoACollectionPhase1Strip, pixelTrack::TracksHostPhase1Strip);

#endif  // DataFormats_TrackSoA_interface_alpaka_TracksSoACollection_h
