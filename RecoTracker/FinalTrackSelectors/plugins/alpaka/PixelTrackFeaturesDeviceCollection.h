#ifndef RecoTracker_FinalTrackSelectors_alpaka_PixelTrackFeaturesDeviceCollection_h
#define RecoTracker_FinalTrackSelectors_alpaka_PixelTrackFeaturesDeviceCollection_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PixelTrackScoresOnDevice = PortableCollection<PixelTrackScoresSoA>;
  // Two-block features collection, constructed with the per-block sizes (nFit, nHit): the hit
  // block is sized 0 when no model consumes it and then costs no memory.
  using PixelTrackFeaturesOnDevice = PortableCollection<PixelTrackFeaturesBlocksSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
