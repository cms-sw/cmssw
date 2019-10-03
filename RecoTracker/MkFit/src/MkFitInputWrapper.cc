#include "RecoTracker/MkFit/interface/MkFitInputWrapper.h"

// mkFit includes
#include "Hit.h"
#include "LayerNumberConverter.h"
#include "Track.h"

MkFitInputWrapper::MkFitInputWrapper() = default;

MkFitInputWrapper::MkFitInputWrapper(MkFitHitIndexMap hitIndexMap,
                                     std::vector<mkfit::HitVec> hits,
                                     mkfit::TrackVec seeds)
    : hitIndexMap_{std::move(hitIndexMap)},
      hits_{std::move(hits)},
      seeds_{std::make_unique<mkfit::TrackVec>(std::move(seeds))} {}

MkFitInputWrapper::~MkFitInputWrapper() = default;

MkFitInputWrapper::MkFitInputWrapper(MkFitInputWrapper&&) = default;
MkFitInputWrapper& MkFitInputWrapper::operator=(MkFitInputWrapper&&) = default;
