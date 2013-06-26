#ifndef CommonTools_ParticleFlow_MuonIDPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_MuonIDPFCandidateSelectorDefinition

/**
   \class    pf2pat::MuonIDPFCandidateSelectorDefinition MuonIDPFCandidateSelectorDefinition.h "CommonTools/ParticleFlow/interface/MuonIDPFCandidateSelectorDefinition.h"
   \brief    Selects PFCandidates basing on cuts provided with string cut parser

   \author   Giovanni Petrucciani
   \version  $Id: MuonIDPFCandidateSelectorDefinition.h,v 1.2 2011/04/06 12:12:38 rwolf Exp $
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"

namespace pf2pat {

  struct MuonIDPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {
    
    MuonIDPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
      muonCut_( cfg.getParameter< std::string >( "cut" ) )
    { 
    }

    void select( const HandleToCollection & hc, 
		 const edm::Event & e,
		 const edm::EventSetup& s) {
      selected_.clear();

      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin(); 
	   pfc != hc->end(); ++pfc, ++key) {

        reco::MuonRef muR = pfc->muonRef();

        // skip ones without a ref to a reco::Muon: they won't be matched anyway	
        if (muR.isNull()) continue;

        // convert into a pat::Muon, so that the 'muonID' method is available
        pat::Muon patMu(*muR);

        // apply muon id
        if (muonCut_(patMu)) {
            selected_.push_back( reco::PFCandidate(*pfc) );
            reco::PFCandidatePtr ptrToMother( hc, key );
            selected_.back().setSourceCandidatePtr( ptrToMother );
        }
      }
    }
    
    private:
        StringCutObjectSelector<pat::Muon> muonCut_;
  };
}

#endif
