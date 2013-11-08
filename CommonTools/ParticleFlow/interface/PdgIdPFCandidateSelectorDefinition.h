#ifndef CommonTools_ParticleFlow_PdgIdPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_PdgIdPFCandidateSelectorDefinition

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"

namespace pf2pat {

  class PdgIdPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

  public:
    PdgIdPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
      pdgIds_( cfg.getParameter< std::vector<int> >( "pdgId" ) ) { }

    void select( const HandleToCollection & hc,
		 const edm::EventBase & e,
		 const edm::EventSetup& s) {
      selected_.clear();

      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin();
	   pfc != hc->end(); ++pfc, ++key) {

	for(unsigned iId=0; iId<pdgIds_.size(); iId++) {
	  if ( pfc->pdgId() == pdgIds_[iId] ) {
	    selected_.push_back( reco::PFCandidate(*pfc) );
	    reco::PFCandidatePtr ptrToMother( hc, key );
	    selected_.back().setSourceCandidatePtr( ptrToMother );
	    break;
	  }
	}
      }
    }

    private:
    std::vector<int> pdgIds_;
  };

}

#endif
