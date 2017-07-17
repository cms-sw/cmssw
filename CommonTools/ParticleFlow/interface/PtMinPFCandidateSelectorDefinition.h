#ifndef CommonTools_ParticleFlow_PtMinPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_PtMinPFCandidateSelectorDefinition

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"

namespace pf2pat {

  class PtMinPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

  public:
    PtMinPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
      ptMin_( cfg.getParameter< double >( "ptMin" ) ) { }


    void select( const HandleToCollection & hc,
		 const edm::EventBase & e,
		 const edm::EventSetup& s
		 ) {
      selected_.clear();

      assert( hc.isValid() );


      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin();
	   pfc != hc->end(); ++pfc, ++key) {

 	if( pfc->pt() > ptMin_ ) {
	  selected_.push_back( reco::PFCandidate(*pfc) );
	  reco::PFCandidatePtr ptrToMother( hc, key );
	  selected_.back().setSourceCandidatePtr( ptrToMother );

	}
      }
    }

/*     const container& selected() const {return selected_;} */

    private:
      double ptMin_;
  };

}

#endif
