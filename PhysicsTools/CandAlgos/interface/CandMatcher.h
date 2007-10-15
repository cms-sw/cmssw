#ifndef CandAlgos_CandMatcher_h
#define CandAlgos_CandMatcher_h
/* \class CandMatcher
 *
 * Producer fo simple Candidate match map
 *
 */
#include "PhysicsTools/UtilAlgos/interface/Matcher.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  namespace modules {
   
    template<typename S, typename D = DeltaR<reco::Candidate> >
    class CandMatcher : public Matcher<reco::CandidateCollection, reco::CandidateCollection, S, D> {
    public:
      CandMatcher(  const edm::ParameterSet & cfg ) : 
	Matcher<reco::CandidateCollection, reco::CandidateCollection, S, D>( cfg ) { }
      ~CandMatcher() { }
    };

  }
}
#endif
