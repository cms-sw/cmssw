#ifndef Candidate_OverlapChecker_h
#define Candidate_OverlapChecker_h
/** \class OverlapChecker
 *
 * Functor that checks the overlap of two Candidate objects
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MuonCandidateProducer.h,v 1.2 2006/03/03 13:40:21 llista Exp $
 *
 */

namespace reco {
  class Candidate; 
}

class OverlapChecker {
public:
  /// return true if two candidates overlap
  bool operator()( const reco::Candidate &, const reco::Candidate & ) const;
};

#endif
