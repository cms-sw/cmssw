#ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
#define CUDADataFormats_Track_PixelTrackHeterogeneous_h

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

template <typename TrackerTraits>
using PixelTrackHeterogeneousT = HeterogeneousSoA<pixelTrack::TrackSoAT<TrackerTraits>>;

#endif  // #ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
