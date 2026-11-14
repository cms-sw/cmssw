#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Track.h"

MkFitOutputWrapper::MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer)
    : tracks_{std::move(tracks)}, propagatedToFirstLayer_{propagatedToFirstLayer} {}

MkFitOutputWrapper::MkFitOutputWrapper(mkfit::TrackVec tracks,
                                       bool propagatedToFirstLayer,
                                       mkfit::FinalFitHitStates hitStates)
    : tracks_{std::move(tracks)}, propagatedToFirstLayer_{propagatedToFirstLayer}, hitStates_{std::move(hitStates)} {}

MkFitOutputWrapper::~MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(MkFitOutputWrapper&&) = default;
MkFitOutputWrapper& MkFitOutputWrapper::operator=(MkFitOutputWrapper&&) = default;
