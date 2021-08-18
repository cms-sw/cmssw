#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"

// mkFit includes
#include "Track.h"

MkFitOutputWrapper::MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer)
    : tracks_{std::move(tracks)}, propagatedToFirstLayer_{propagatedToFirstLayer} {}

MkFitOutputWrapper::~MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(MkFitOutputWrapper&&) = default;
MkFitOutputWrapper& MkFitOutputWrapper::operator=(MkFitOutputWrapper&&) = default;
