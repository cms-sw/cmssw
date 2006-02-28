#ifndef Candidate_OverlapChecker_h
#define Candidate_OverlapChecker_h
// $Id: OverlapChecker.h,v 1.3 2006/02/21 10:37:32 llista Exp $
namespace reco {
  class Candidate; 
}

class OverlapChecker {
public:
  OverlapChecker() { }
  bool operator()( const reco::Candidate &, const reco::Candidate & ) const;
};

#endif
