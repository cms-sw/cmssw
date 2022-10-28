#ifndef CandAlgos_CandMatcher_h
#define CandAlgos_CandMatcher_h
/* \class CandMatcher
 *
 * Producer fo simple Candidate match map
 *
 */
#include "CommonTools/UtilAlgos/interface/Matcher.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/CandUtils/interface/CandMapTrait.h"

namespace reco::modules {
  template <typename S, typename Collection = CandidateCollection, typename D = DeltaR<reco::Candidate> >
  using CandMatcher = Matcher<Collection, Collection, S, D, typename reco::helper::CandMapTrait<Collection>::type>;
}  // namespace reco::modules
#endif
