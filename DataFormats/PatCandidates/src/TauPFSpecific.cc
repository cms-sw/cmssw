#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"

#include "DataFormats/JetReco/interface/Jet.h"

pat::tau::TauPFSpecific::TauPFSpecific(const reco::PFTau& tau) :
    // reference to PFJet from which PFTau was made
    pfJetRef_(tau.jetRef()),
    // Leading track/charged candidate
    leadPFChargedHadrCand_(tau.leadPFChargedHadrCand()),    
    leadPFChargedHadrCandsignedSipt_(tau.leadPFChargedHadrCandsignedSipt()),
    // Leading neutral candidate
    leadPFNeutralCand_(tau.leadPFNeutralCand()), 
    // Leading charged or neutral candidate
    leadPFCand_(tau.leadPFCand()), 
    // Signal cone
    selectedSignalPFCands_(tau.signalPFCands()), 
    selectedSignalPFChargedHadrCands_(tau.signalPFChargedHadrCands()), 
    selectedSignalPFNeutrHadrCands_(tau.signalPFNeutrHadrCands()), 
    selectedSignalPFGammaCands_(tau.signalPFGammaCands()),
    signalTauChargedHadronCandidates_(tau.signalTauChargedHadronCandidates()),
    signalPiZeroCandidates_(tau.signalPiZeroCandidates()),
    // Isolation cone
    selectedIsolationPFCands_(tau.isolationPFCands()), 
    selectedIsolationPFChargedHadrCands_(tau.isolationPFChargedHadrCands()), 
    selectedIsolationPFNeutrHadrCands_(tau.isolationPFNeutrHadrCands()), 
    selectedIsolationPFGammaCands_(tau.isolationPFGammaCands()),
    isolationTauChargedHadronCandidates_(tau.isolationTauChargedHadronCandidates()),
    isolationPiZeroCandidates_(tau.isolationPiZeroCandidates()),
    isolationPFChargedHadrCandsPtSum_(tau.isolationPFChargedHadrCandsPtSum()),    
    isolationPFGammaCandsEtSum_(tau.isolationPFGammaCandsEtSum()),
    // Other useful variables 
    maximumHCALPFClusterEt_(tau.maximumHCALPFClusterEt()),
    emFraction_(tau.emFraction()),
    hcalTotOverPLead_(tau.hcalTotOverPLead()),
    hcalMaxOverPLead_(tau.hcalMaxOverPLead()),
    hcal3x3OverPLead_(tau.hcal3x3OverPLead()),
    ecalStripSumEOverPLead_(tau.ecalStripSumEOverPLead()),
    bremsRecoveryEOverPLead_(tau.bremsRecoveryEOverPLead()),
    // Electron rejection variables
    electronPreIDTrack_(tau.electronPreIDTrack()),
    electronPreIDOutput_(tau.electronPreIDOutput()),
    electronPreIDDecision_(tau.electronPreIDDecision()),
    // Muon rejection variables
    caloComp_(tau.caloComp()),
    segComp_(tau.segComp()),
    muonDecision_(tau.muonDecision()),
    dxy_(0.),
    dxy_error_(1.e+3),
    hasSV_(false)
{
  if ( tau.jetRef().isAvailable() && tau.jetRef().isNonnull() ) { // CV: add protection to ease transition to new CMSSW 4_2_x RecoTauTags
    p4Jet_ = tau.jetRef()->p4();
    reco::Jet::EtaPhiMoments etaPhiStatistics = tau.jetRef()->etaPhiStatistics();
    etaetaMoment_ = etaPhiStatistics.etaEtaMoment;
    phiphiMoment_ = etaPhiStatistics.phiPhiMoment;
    etaphiMoment_ = etaPhiStatistics.etaPhiMoment;
  }
}
