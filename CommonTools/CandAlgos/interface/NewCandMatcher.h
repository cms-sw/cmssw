#ifndef CandAlgos_NewCandMatcher_h
#define CandAlgos_NewCandMatcher_h
/* \class CandMatcher
 *
 * Producer fo simple Candidate match map
 *
 */
#include "CommonTools/UtilAlgos/interface/NewMatcher.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  namespace modulesNew {
    template<typename S, typename C1, typename C2, typename D = DeltaR<reco::Candidate> >
    class CandMatcher : public Matcher<C1, C2, S, D> {
    public:
      CandMatcher(const edm::ParameterSet & cfg ) : Matcher<C1, C2, S, D>( cfg ) { }
      ~CandMatcher() { }
    };

  }
}
#endif
