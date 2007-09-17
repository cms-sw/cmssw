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
    namespace helper {
      template<typename T>
      struct CandMapTrait {
      };

      template<>
      struct CandMapTrait<CandidateCollection> {
	typedef edm::AssociationMap<edm::OneToOne<CandidateCollection, CandidateCollection> > type;
      };

      template<>
      struct CandMapTrait<CandidateView> {
	typedef edm::AssociationMap<edm::OneToOneGeneric<CandidateView, CandidateView> > type;
      };
    }
   
    template<typename S, typename Collection = CandidateCollection, typename D = DeltaR<reco::Candidate> >
    class CandMatcher : 
      public Matcher<Collection, Collection, S, D, typename helper::CandMapTrait<Collection>::type> {
      public:
        CandMatcher(  const edm::ParameterSet & cfg ) : 
          Matcher<Collection, Collection, S, D, typename helper::CandMapTrait<Collection>::type>( cfg ) { }
      ~CandMatcher() { }
    };

  }
}
#endif
