#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

MkFitHitWrapper::MkFitHitWrapper() = default;
MkFitHitWrapper::~MkFitHitWrapper() = default;

MkFitHitWrapper::MkFitHitWrapper(MkFitHitWrapper&&) = default;
MkFitHitWrapper& MkFitHitWrapper::operator=(MkFitHitWrapper&&) = default;
