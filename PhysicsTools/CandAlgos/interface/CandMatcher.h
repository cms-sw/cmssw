#ifndef CandAlgos_CandMatcher_h
#define CandAlgos_CandMatcher_h
/* \class CandMatcher
 *
 * Producer fo simple Candidate match map
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/UtilAlgos/interface/DeltaR.h"

namespace reco {
  namespace modules {
    class CandMatcherBase : public edm::EDProducer {
    public:
      CandMatcherBase( const edm::ParameterSet & );
      ~CandMatcherBase();
    private:
      void produce( edm::Event&, const edm::EventSetup& );
      edm::InputTag src_;
      edm::InputTag matched_;
      double distMin_;
      virtual double matchDistance( const reco::Candidate &, const reco::Candidate & ) const = 0;
      virtual bool select( const reco::Candidate &, const reco::Candidate & ) const = 0;
    };
  }
}


namespace reco {
  namespace modules {
    
    template<typename S, typename D = DeltaR<reco::Candidate> >
    class CandMatcher : public CandMatcherBase {
    public:
      CandMatcher(  const edm::ParameterSet & cfg ) : 
	CandMatcherBase( cfg ),
        select_( reco::modules::make<S>( cfg ) ), 
	distance_( reco::modules::make<D>( cfg ) ) { }
      ~CandMatcher() { }
    private:
      double matchDistance( const reco::Candidate & c1, const reco::Candidate & c2 ) const {
	return distance_( c1, c2 );
      }
      bool select( const reco::Candidate & c1, const reco::Candidate & c2 ) const { 
	return select_( c1, c2 ); 
      }
      S select_;
      D distance_;
    };
  }
}
#endif
