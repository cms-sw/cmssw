#ifndef DataFormats_Track_TracksHost_H
#define DataFormats_Track_TracksHost_H

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

// TODO: The class is created via inheritance of the PortableHostCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits>
class TracksHost : public PortableHostCollection<reco::TrackLayout<TrackerTraits>> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  TracksHost() = default;  // Needed for the dictionary; not sure if line above is needed anymore

  using PortableHostCollection<reco::TrackLayout<TrackerTraits>>::view;
  using PortableHostCollection<reco::TrackLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<reco::TrackLayout<TrackerTraits>>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TracksHost<TrackerTraits>(TQueue& queue)
      : PortableHostCollection<reco::TrackLayout<TrackerTraits>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit TracksHost(alpaka_common::DevHost const& host)
      : PortableHostCollection<reco::TrackLayout<TrackerTraits>>(S, host) {}
};

namespace pixelTrack {

  using TracksHostPhase1 = TracksHost<pixelTopology::Phase1>;
  using TracksHostPhase2 = TracksHost<pixelTopology::Phase2>;
  using TracksHostHIonPhase1 = TracksHost<pixelTopology::HIonPhase1>;

}  // namespace pixelTrack

#endif  // DataFormats_Track_TracksHost_H
