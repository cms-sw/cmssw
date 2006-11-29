#ifndef CandUtils_CandMatcher_h
#define CandUtils_CandMatcher_h
/* class CandMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include <set>

class CandMatcher {
public:
  /// constructor
  explicit CandMatcher( const reco::CandMatchMap & map );
  /// get match from transient reference
  reco::CandidateRef operator()( const reco::Candidate & ) const;

private:
  /// reference to match map, typically taken from the event
  const reco::CandMatchMap & map_;
  /// pointer map type
  typedef std::map<const reco::Candidate *, reco::CandidateRef> RefMap;
  /// pointer map of candidates (e.g.: reco)
  RefMap candRefs_;
  /// pointer map of matched candidates (e.g.: MC truth)
  RefMap matchedRefs_;
  /// mother + n.daughters indices from matched
  std::vector<std::vector<size_t> > matchedMothers_;
};

#endif
