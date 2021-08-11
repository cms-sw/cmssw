#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"

// mkFit includes
#include "Track.h"

MkFitSeedWrapper::MkFitSeedWrapper() = default;

MkFitSeedWrapper::MkFitSeedWrapper(mkfit::TrackVec seeds)
    : seeds_{std::make_unique<mkfit::TrackVec>(std::move(seeds))} {}

MkFitSeedWrapper::~MkFitSeedWrapper() = default;

MkFitSeedWrapper::MkFitSeedWrapper(MkFitSeedWrapper&&) = default;
MkFitSeedWrapper& MkFitSeedWrapper::operator=(MkFitSeedWrapper&&) = default;
