#ifndef Candidate_OverlapChecker_h
#define Candidate_OverlapChecker_h
/** \class OverlapChecker
 *
 * Functor that checks the overlap of two Candidate objects
 *
 * \author Luca Lista, INFN
 *
 *
 *
 */

namespace reco {
  class Candidate;
}

class OverlapChecker {
public:
  /// return true if two candidates overlap
  bool operator()(const reco::Candidate &, const reco::Candidate &) const;
};

#endif
