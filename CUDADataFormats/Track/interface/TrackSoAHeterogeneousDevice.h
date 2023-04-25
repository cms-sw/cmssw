#ifndef CUDADataFormats_Track_TrackHeterogeneousDevice_H
#define CUDADataFormats_Track_TrackHeterogeneousDevice_H

#include <cstdint>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits>
class TrackSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<TrackLayout<TrackerTraits>> {
public:
  using cms::cuda::PortableDeviceCollection<TrackLayout<TrackerTraits>>::view;
  using cms::cuda::PortableDeviceCollection<TrackLayout<TrackerTraits>>::const_view;
  using cms::cuda::PortableDeviceCollection<TrackLayout<TrackerTraits>>::buffer;
  using cms::cuda::PortableDeviceCollection<TrackLayout<TrackerTraits>>::bufferSize;

  TrackSoAHeterogeneousDevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousDevice(cudaStream_t stream)
      : cms::cuda::PortableDeviceCollection<TrackLayout<TrackerTraits>>(TrackerTraits::maxNumberOfTuples, stream) {}
};

namespace pixelTrack {

  using TrackSoADevicePhase1 = TrackSoAHeterogeneousDevice<pixelTopology::Phase1>;
  using TrackSoADevicePhase2 = TrackSoAHeterogeneousDevice<pixelTopology::Phase2>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
