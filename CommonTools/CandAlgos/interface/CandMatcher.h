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

namespace reco {
  namespace modules {
    template<typename S, typename Collection = CandidateCollection, typename D = DeltaR<reco::Candidate> >
    class CandMatcher : 
      public Matcher<Collection, Collection, S, D, typename reco::helper::CandMapTrait<Collection>::type> {
      public:
        CandMatcher(  const edm::ParameterSet & cfg ) : 
          Matcher<Collection, Collection, S, D, typename reco::helper::CandMapTrait<Collection>::type>( cfg ) { }
      ~CandMatcher() { }
    };

  }
}
#endif
