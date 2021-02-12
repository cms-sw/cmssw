#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"

// mkFit includes
#include "Hit.h"
#include "mkFit/HitStructures.h"

MkFitHitWrapper::MkFitHitWrapper() = default;
MkFitHitWrapper::MkFitHitWrapper(mkfit::TrackerInfo const& trackerInfo)
    : eventOfHits_(std::make_unique<mkfit::EventOfHits>(trackerInfo)),
      pixelHits_(std::make_unique<mkfit::HitVec>()),
      outerHits_(std::make_unique<mkfit::HitVec>()) {}

MkFitHitWrapper::~MkFitHitWrapper() = default;

MkFitHitWrapper::MkFitHitWrapper(MkFitHitWrapper&&) = default;
MkFitHitWrapper& MkFitHitWrapper::operator=(MkFitHitWrapper&&) = default;
