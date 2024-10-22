//
//

#ifndef DataFormats_PatCandidates_Tau_PFSpecific_h
#define DataFormats_PatCandidates_Tau_PFSpecific_h

/**
  \class    pat::tau::PFSpecific TauPFSpecific.h "DataFormats/PatCandidates/interface/TauPFSpecific.h"
  \brief    Structure to hold information specific to a PFTau made from PFCandidates inside a pat::Tau

  \author   Giovanni Petrucciani
*/

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace pat {
  namespace tau {

    struct TauPFSpecific {
      // dummy constructor for ROOT I/O
      TauPFSpecific() {}
      // constructor from PFTau
      TauPFSpecific(const reco::PFTau& tau);
      // datamembers
      reco::JetBaseRef pfJetRef_;
      reco::CandidatePtr leadPFChargedHadrCand_;
      float leadPFChargedHadrCandsignedSipt_;
      reco::PFCandidatePtr leadPFNeutralCand_;
      reco::PFCandidatePtr leadPFCand_;
      std::vector<reco::PFCandidatePtr> selectedSignalPFCands_;
      std::vector<reco::PFCandidatePtr> selectedSignalPFChargedHadrCands_;
      std::vector<reco::PFCandidatePtr> selectedSignalPFNeutrHadrCands_;
      std::vector<reco::PFCandidatePtr> selectedSignalPFGammaCands_;
      std::vector<reco::PFRecoTauChargedHadron> signalTauChargedHadronCandidates_;
      std::vector<reco::RecoTauPiZero> signalPiZeroCandidates_;
      std::vector<reco::PFCandidatePtr> selectedIsolationPFCands_;
      std::vector<reco::PFCandidatePtr> selectedIsolationPFChargedHadrCands_;
      std::vector<reco::PFCandidatePtr> selectedIsolationPFNeutrHadrCands_;
      std::vector<reco::PFCandidatePtr> selectedIsolationPFGammaCands_;
      std::vector<reco::PFRecoTauChargedHadron> isolationTauChargedHadronCandidates_;
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

      float etaetaMoment_;
      float phiphiMoment_;
      float etaphiMoment_;

      float bendCorrMass_;
      float signalConeSize_;
    };

  }  // namespace tau
}  // namespace pat

#endif
