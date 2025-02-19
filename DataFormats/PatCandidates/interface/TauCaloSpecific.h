//
// $Id: TauCaloSpecific.h,v 1.3 2011/07/21 16:42:41 veelken Exp $
//

#ifndef DataFormats_PatCandidates_Tau_CaloSpecific_h
#define DataFormats_PatCandidates_Tau_CaloSpecific_h

/**
  \class    pat::tau::CaloSpecific TauCaloSpecific.h "DataFormats/PatCandidates/interface/TauCaloSpecific.h"
  \brief    Structure to hold information specific to a CaloTau inside a pat::Tau

  \author   Giovanni Petrucciani
  \version  $Id: TauCaloSpecific.h,v 1.3 2011/07/21 16:42:41 veelken Exp $
*/

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pat { namespace tau {

struct TauCaloSpecific {
// dummy constructor for ROOT I/O
    TauCaloSpecific() {}
// constructor from CaloTau
    TauCaloSpecific(const reco::CaloTau &tau) ;
// datamembers 
    reco::CaloTauTagInfoRef CaloTauTagInfoRef_;
    float leadTracksignedSipt_;
    float leadTrackHCAL3x3hitsEtSum_;
    float leadTrackHCAL3x3hottesthitDEta_;
    float signalTracksInvariantMass_;
    float TracksInvariantMass_; 
    float isolationTracksPtSum_;
    float isolationECALhitsEtSum_;
    float maximumHCALhitEt_;
    reco::Candidate::LorentzVector p4Jet_;
    float etaetaMoment_;
    float phiphiMoment_;
    float etaphiMoment_;
};

} }

#endif
