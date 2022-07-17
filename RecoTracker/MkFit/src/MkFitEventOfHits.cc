#include "RecoTracker/MkFit/interface/MkFitEventOfHits.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

MkFitEventOfHits::MkFitEventOfHits() = default;
MkFitEventOfHits::MkFitEventOfHits(std::unique_ptr<mkfit::EventOfHits> eoh) : eventOfHits_(std::move(eoh)) {}
MkFitEventOfHits::~MkFitEventOfHits() = default;

MkFitEventOfHits::MkFitEventOfHits(MkFitEventOfHits&&) = default;
MkFitEventOfHits& MkFitEventOfHits::operator=(MkFitEventOfHits&&) = default;
