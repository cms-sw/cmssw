#ifndef CandAlgos_NewCandMatcher_h
#define CandAlgos_NewCandMatcher_h
/* \class CandMatcher
 *
 * Producer fo simple Candidate match map
 *
 */
#include "CommonTools/UtilAlgos/interface/NewMatcher.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco::modulesNew {
  template <typename S, typename C1, typename C2, typename D = DeltaR<reco::Candidate> >
  using CandMatcher = Matcher<C1, C2, S, D>;
}  // namespace reco::modulesNew
#endif
