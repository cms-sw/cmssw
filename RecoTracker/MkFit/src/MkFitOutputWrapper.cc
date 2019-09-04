#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"

// mkFit includes
#include "Track.h"

MkFitOutputWrapper::MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(mkfit::TrackVec&& candidateTracks, mkfit::TrackVec&& fitTracks)
    : candidateTracks_{std::move(candidateTracks)}, fitTracks_{std::move(fitTracks)} {}

MkFitOutputWrapper::~MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(MkFitOutputWrapper&&) = default;
MkFitOutputWrapper& MkFitOutputWrapper::operator=(MkFitOutputWrapper&&) = default;
