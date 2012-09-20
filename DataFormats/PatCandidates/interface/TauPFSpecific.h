//
// $Id: TauPFSpecific.h,v 1.6 2011/07/21 16:42:41 veelken Exp $
//

#ifndef DataFormats_PatCandidates_Tau_PFSpecific_h
#define DataFormats_PatCandidates_Tau_PFSpecific_h

/**
  \class    pat::tau::PFSpecific TauPFSpecific.h "DataFormats/PatCandidates/interface/TauPFSpecific.h"
  \brief    Structure to hold information specific to a PFTau inside a pat::Tau

  \author   Giovanni Petrucciani
  \version  $Id: TauPFSpecific.h,v 1.6 2011/07/21 16:42:41 veelken Exp $
*/

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pat { namespace tau {

struct TauPFSpecific {
// dummy constructor for ROOT I/O
  TauPFSpecific() {}
// constructor from PFTau
  TauPFSpecific(const reco::PFTau& tau);
// datamembers 
  reco::PFJetRef pfJetRef_;
  reco::PFCandidateRef leadPFChargedHadrCand_;
  float leadPFChargedHadrCandsignedSipt_;
  reco::PFCandidateRef leadPFNeutralCand_;
  reco::PFCandidateRef leadPFCand_;
  reco::PFCandidateRefVector selectedSignalPFCands_;
  reco::PFCandidateRefVector selectedSignalPFChargedHadrCands_;
  reco::PFCandidateRefVector selectedSignalPFNeutrHadrCands_;
  reco::PFCandidateRefVector selectedSignalPFGammaCands_;
  std::vector<reco::RecoTauPiZero> signalPiZeroCandidates_;
  reco::PFCandidateRefVector selectedIsolationPFCands_;
  reco::PFCandidateRefVector selectedIsolationPFChargedHadrCands_;
  reco::PFCandidateRefVector selectedIsolationPFNeutrHadrCands_;
  reco::PFCandidateRefVector selectedIsolationPFGammaCands_;
  std::vector<reco::RecoTauPiZero> isolationPiZeroCandidates_;
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
  
  reco::Candidate::LorentzVector p4Jet_;
  float etaetaMoment_;
  float phiphiMoment_;
  float etaphiMoment_;
  
  int decayMode_;
};

} }

#endif
