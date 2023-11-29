#ifndef DataFormats_Track_TrackSoAHost_H
#define DataFormats_Track_TrackSoAHost_H

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/TrackSoA/interface/TrackLayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

// TODO: The class is created via inheritance of the PortableHostCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits>
class TrackSoAHost : public PortableHostCollection<TrackLayout<TrackerTraits>> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  TrackSoAHost() = default;  // Needed for the dictionary; not sure if line above is needed anymore

  using PortableHostCollection<TrackLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackLayout<TrackerTraits>>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackSoAHost<TrackerTraits>(TQueue queue) : PortableHostCollection<TrackLayout<TrackerTraits>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit TrackSoAHost(alpaka_common::DevHost const& host)
      : PortableHostCollection<TrackLayout<TrackerTraits>>(S, host) {}
};

namespace pixelTrack {

  using TrackSoAHostPhase1 = TrackSoAHost<pixelTopology::Phase1>;
  using TrackSoAHostPhase2 = TrackSoAHost<pixelTopology::Phase2>;
  using TrackSoAHostHIonPhase1 = TrackSoAHost<pixelTopology::HIonPhase1>;

}  // namespace pixelTrack

#endif  // DataFormats_Track_TrackSoAHost_H
