#ifndef DataFormats_Track_interface_TrackSoADevice_h
#define DataFormats_Track_interface_TrackSoADevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/TrackSoA/interface/TrackLayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits, typename TDev>
class TrackSoADevice : public PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  TrackSoADevice() = default;                                     // necessary for ROOT dictionaries

  using PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackSoADevice<TrackerTraits, TDev>(TQueue queue)
      : PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>(S, queue) {}
};

namespace pixelTrack {

  template <typename TDev>
  using TrackSoADevicePhase1 = TrackSoADevice<pixelTopology::Phase1, TDev>;
  template <typename TDev>
  using TrackSoADevicePhase2 = TrackSoADevice<pixelTopology::Phase2, TDev>;

}  // namespace pixelTrack

#endif  // DataFormats_Track_TrackSoADevice_H
