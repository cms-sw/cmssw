#ifndef CANDIDATES_CANDIDATE_UTILS_H
#define CANDIDATES_CANDIDATE_UTILS_H

#include "DataFormats/Candidate/interface/Candidate.h"

class TEveTrack;
class TEveTrackPropagator;
class TEveStraightLineSet;

namespace fireworks {

  TEveTrack* prepareCandidate(const reco::Candidate& track, TEveTrackPropagator* propagator);

  void addStraightLineSegment(TEveStraightLineSet* marker, reco::Candidate const* cand, double scale_factor = 2);
}  // namespace fireworks

#endif  // CANDIDATES_CANDIDATE_UTILS_H
