#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"

#include "DataFormats/JetReco/interface/Jet.h"

pat::tau::TauPFSpecific::TauPFSpecific(const reco::PFTau &tau) :
    // Tau tag ingo
    //PFTauTagInfoRef_(tau.pfTauTagInfoRef()),
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
    // Isolation cone
    selectedIsolationPFCands_(tau.isolationPFCands()), 
    selectedIsolationPFChargedHadrCands_(tau.isolationPFChargedHadrCands()), 
    selectedIsolationPFNeutrHadrCands_(tau.isolationPFNeutrHadrCands()), 
    selectedIsolationPFGammaCands_(tau.isolationPFGammaCands()),
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
    muonDecision_(tau.muonDecision())
{
  //reco::Jet::EtaPhiMoments etaPhiStatistics = tau.pfTauTagInfoRef()->pfjetRef()->etaPhiStatistics();
  //etaetaMoment_ = etaPhiStatistics.etaEtaMoment;
  //phiphiMoment_ = etaPhiStatistics.phiPhiMoment;
  //etaphiMoment_ = etaPhiStatistics.etaPhiMoment;
}
