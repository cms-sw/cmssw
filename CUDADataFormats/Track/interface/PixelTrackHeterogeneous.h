#ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
#define CUDADataFormats_Track_PixelTrackHeterogeneous_h

#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"

using PixelTrackHeterogeneous = memoryPool::Buffer<pixelTrack::TrackSoA>;

#endif  // #ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
