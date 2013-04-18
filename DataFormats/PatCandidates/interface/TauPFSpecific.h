//
// $Id: TauPFSpecific.h,v 1.4 2010/02/03 10:31:54 veelken Exp $
//

#ifndef DataFormats_PatCandidates_Tau_PFSpecific_h
#define DataFormats_PatCandidates_Tau_PFSpecific_h

/**
  \class    pat::tau::PFSpecific TauPFSpecific.h "DataFormats/PatCandidates/interface/TauPFSpecific.h"
  \brief    Structure to hold information specific to a PFTau inside a pat::Tau

  \author   Giovanni Petrucciani
  \version  $Id: TauPFSpecific.h,v 1.4 2010/02/03 10:31:54 veelken Exp $
*/

#include "DataFormats/TauReco/interface/PFTau.h"

namespace pat { namespace tau {

struct TauPFSpecific {
// dummy constructor for ROOT I/O
    TauPFSpecific() {}
// constructor from PFTau
    TauPFSpecific(const reco::PFTau &tau) ;
// datamembers 
  //reco::PFTauTagInfoRef PFTauTagInfoRef_;
    reco::PFCandidateRef leadPFChargedHadrCand_;
    float leadPFChargedHadrCandsignedSipt_;
    reco::PFCandidateRef leadPFNeutralCand_;
    reco::PFCandidateRef leadPFCand_;
    reco::PFCandidateRefVector selectedSignalPFCands_, selectedSignalPFChargedHadrCands_, selectedSignalPFNeutrHadrCands_, selectedSignalPFGammaCands_;
    reco::PFCandidateRefVector selectedIsolationPFCands_, selectedIsolationPFChargedHadrCands_, selectedIsolationPFNeutrHadrCands_, selectedIsolationPFGammaCands_;
    float isolationPFChargedHadrCandsPtSum_;
    float isolationPFGammaCandsEtSum_;
    float maximumHCALPFClusterEt_;
    
    float emFraction_;
    float hcalTotOverPLead_;
    float hcalMaxOverPLead_;
    float hcal3x3OverPLead_;
    float ecalStripSumEOverPLead_;
    float bremsRecoveryEOverPLead_;
    reco::TrackRef electronPreIDTrack_;
    float electronPreIDOutput_;
    bool electronPreIDDecision_;

    float caloComp_;
    float segComp_;
    bool muonDecision_;

    float etaetaMoment_;
    float phiphiMoment_;
    float etaphiMoment_;

    int decayMode_;
};

} }

#endif
