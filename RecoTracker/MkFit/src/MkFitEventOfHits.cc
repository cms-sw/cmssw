#include "RecoTracker/MkFit/interface/MkFitEventOfHits.h"

// mkFit includes
#include "mkFit/HitStructures.h"

MkFitEventOfHits::MkFitEventOfHits() = default;
MkFitEventOfHits::MkFitEventOfHits(std::unique_ptr<mkfit::EventOfHits> eoh) : eventOfHits_(std::move(eoh)) {}
MkFitEventOfHits::~MkFitEventOfHits() = default;

MkFitEventOfHits::MkFitEventOfHits(MkFitEventOfHits&&) = default;
MkFitEventOfHits& MkFitEventOfHits::operator=(MkFitEventOfHits&&) = default;
