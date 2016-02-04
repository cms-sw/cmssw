#include "DataFormats/PatCandidates/interface/TauCaloSpecific.h"

pat::tau::TauCaloSpecific::TauCaloSpecific(const reco::CaloTau &tau) :
    CaloTauTagInfoRef_(tau.caloTauTagInfoRef()),
    leadTracksignedSipt_(tau.leadTracksignedSipt()),
    leadTrackHCAL3x3hitsEtSum_(tau.leadTrackHCAL3x3hitsEtSum()),
    leadTrackHCAL3x3hottesthitDEta_(tau.leadTrackHCAL3x3hottesthitDEta()),
    signalTracksInvariantMass_(tau.signalTracksInvariantMass()),
    TracksInvariantMass_(tau.TracksInvariantMass()), 
    isolationTracksPtSum_(tau.isolationTracksPtSum()),
    isolationECALhitsEtSum_(tau.isolationECALhitsEtSum()),
    maximumHCALhitEt_(tau.maximumHCALhitEt())
{
  p4Jet_ = tau.caloTauTagInfoRef()->calojetRef()->p4();
  reco::Jet::EtaPhiMoments etaPhiStatistics = tau.caloTauTagInfoRef()->calojetRef()->etaPhiStatistics();
  etaetaMoment_ = etaPhiStatistics.etaEtaMoment;
  phiphiMoment_ = etaPhiStatistics.phiPhiMoment;
  etaphiMoment_ = etaPhiStatistics.etaPhiMoment;
}
