#ifndef CommonTools_ParticleFlow_GenericPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_GenericPFCandidateSelectorDefinition

/**
   \class    pf2pat::GenericPFCandidateSelectorDefinition GenericPFCandidateSelectorDefinition.h "CommonTools/ParticleFlow/interface/GenericPFCandidateSelectorDefinition.h"
   \brief    Selects PFCandidates basing on cuts provided with string cut parser

   \author   Giovanni Petrucciani
   \version  $Id: GenericPFCandidateSelectorDefinition.h,v 1.2 2011/04/06 12:12:38 rwolf Exp $
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace pf2pat {

  struct GenericPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {
    
    GenericPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
      selector_( cfg.getParameter< std::string >( "cut" ) ) { }
    
    void select( const HandleToCollection & hc, 
		 const edm::Event & e,
		 const edm::EventSetup& s) {
      selected_.clear();
      
      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin(); 
	   pfc != hc->end(); ++pfc, ++key) {
	
	if( selector_(*pfc) ) {
	  selected_.push_back( reco::PFCandidate(*pfc) );
	  reco::PFCandidatePtr ptrToMother( hc, key );
	  selected_.back().setSourceCandidatePtr( ptrToMother );
	  
	}
      }
    }
    
    private:
    StringCutObjectSelector<reco::PFCandidate> selector_;
  };
}

#endif
