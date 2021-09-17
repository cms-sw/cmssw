#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"

// mkFit includes
#include "Hit.h"
#include "mkFit/HitStructures.h"

MkFitHitWrapper::MkFitHitWrapper() = default;
MkFitHitWrapper::~MkFitHitWrapper() = default;

MkFitHitWrapper::MkFitHitWrapper(MkFitHitWrapper&&) = default;
MkFitHitWrapper& MkFitHitWrapper::operator=(MkFitHitWrapper&&) = default;
