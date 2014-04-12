#ifndef CommonTools_ParticleFlow_GenericPFJetSelectorDefinition
#define CommonTools_ParticleFlow_GenericPFJetSelectorDefinition


#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CommonTools/ParticleFlow/interface/PFJetSelectorDefinition.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace pf2pat {

  struct GenericPFJetSelectorDefinition : public PFJetSelectorDefinition {

    GenericPFJetSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
      selector_( cfg.getParameter< std::string >( "cut" ) ) { }

    void select( const HandleToCollection & hc,
		 const edm::Event & e,
		 const edm::EventSetup& s) {
      selected_.clear();

      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin();
	   pfc != hc->end(); ++pfc, ++key) {

	if( selector_(*pfc) ) {
	  selected_.push_back( reco::PFJet(*pfc) );
	  reco::CandidatePtr ptrToMother( hc, key );
	  selected_.back().setSourceCandidatePtr( ptrToMother );

	}
      }
    }

    private:
    StringCutObjectSelector<reco::PFJet> selector_;
  };
}

#endif
