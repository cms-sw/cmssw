#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"

// mkFit includes
#include "Hit.h"

MkFitHitWrapper::MkFitHitWrapper() = default;

MkFitHitWrapper::MkFitHitWrapper(MkFitHitIndexMap hitIndexMap, std::vector<mkfit::HitVec> hits)
    : hitIndexMap_{std::move(hitIndexMap)}, hits_{std::move(hits)} {}

MkFitHitWrapper::~MkFitHitWrapper() = default;

MkFitHitWrapper::MkFitHitWrapper(MkFitHitWrapper&&) = default;
MkFitHitWrapper& MkFitHitWrapper::operator=(MkFitHitWrapper&&) = default;
