#ifndef CUDADataFormats_Track_TrackHeterogeneousHost_H
#define CUDADataFormats_Track_TrackHeterogeneousHost_H

#include <cstdint>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"

// TODO: The class is created via inheritance of the PortableHostCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits>
class TrackSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  explicit TrackSoAHeterogeneousHost() : cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>(S) {}

  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::view;
  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::const_view;
  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::buffer;
  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::bufferSize;

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousHost(cudaStream_t stream)
      : cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>(S, stream) {}
};

namespace pixelTrack {

  using TrackSoAHostPhase1 = TrackSoAHeterogeneousHost<pixelTopology::Phase1>;
  using TrackSoAHostPhase2 = TrackSoAHeterogeneousHost<pixelTopology::Phase2>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
