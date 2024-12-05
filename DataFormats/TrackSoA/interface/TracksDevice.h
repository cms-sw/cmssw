#ifndef DataFormats_Track_interface_TracksDevice_h
#define DataFormats_Track_interface_TracksDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits, typename TDev>
class TracksDevice : public PortableDeviceCollection<reco::TrackLayout<TrackerTraits>, TDev> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime

  TracksDevice(edm::Uninitialized)
      : PortableDeviceCollection<reco::TrackLayout<TrackerTraits>, TDev>{edm::kUninitialized} {
  }  // necessary for ROOT dictionaries

  using PortableDeviceCollection<reco::TrackLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<reco::TrackLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<reco::TrackLayout<TrackerTraits>, TDev>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TracksDevice<TrackerTraits, TDev>(TQueue& queue)
      : PortableDeviceCollection<reco::TrackLayout<TrackerTraits>, TDev>(S, queue) {}
};

namespace pixelTrack {

  template <typename TDev>
  using TracksDevicePhase1 = TracksDevice<pixelTopology::Phase1, TDev>;
  template <typename TDev>
  using TracksDeviceHIonPhase1 = TracksDevice<pixelTopology::HIonPhase1, TDev>;
  template <typename TDev>
  using TracksDevicePhase2 = TracksDevice<pixelTopology::Phase2, TDev>;

}  // namespace pixelTrack

#endif  // DataFormats_Track_TracksDevice_H
