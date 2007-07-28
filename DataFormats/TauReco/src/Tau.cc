#include "DataFormats/TauReco/interface/Tau.h"

using namespace reco;

Tau::Tau() {
 TrackRefVector tmp;
  TrackRef leadTk;
  leadTrack_ = leadTk;
  signalTracks_ =tmp ;
  isolationTracks_= tmp;
  selectedTracks_ = tmp;

  PFCandidateRef pfLead;
  leadPFChargedHadrCand_ = pfLead;
  PFCandidateRefVector pfTmp;
  selectedSignalPFChargedHadrCands_ =pfTmp;
  selectedSignalPFNeutrHadrCands_=pfTmp;
  selectedSignalPFGammaCands_ = pfTmp;

  selectedIsolationPFChargedHadrCands_ = pfTmp;
  selectedIsolationPFNeutrHadrCands_ = pfTmp;
  selectedIsolationPFGammaCands_ = pfTmp;

  selectedPFChargedHadrCands_ = pfTmp;
  selectedPFNeutrHadrCands_ = pfTmp;
  selectedPFGammaCands_ = pfTmp;

  maximumHcalTowerEnergy_ = NAN;
  leadPFChargedHadrCandsignedSipt_=NAN;
  leadTracksignedSipt_=NAN;
  mass_ = NAN; 
  trackerMass_ = NAN;
  sumPtIsolation_=NAN;
  emOverHadronEnergy_=NAN;
  emIsolation_=NAN;
  numberOfEcalClusters_=std::numeric_limits<int>::quiet_NaN();
}

Tau::Tau(Charge q, const LorentzVector & p4, const Point & vtx ) :
   RecoCandidate( q, p4, vtx, -15 * q ) {
  TrackRefVector tmp;
  TrackRef leadTk;
  leadTrack_ = leadTk;
  signalTracks_ =tmp ;
  isolationTracks_= tmp;
  selectedTracks_ = tmp;

  PFCandidateRef pfLead;
  leadPFChargedHadrCand_ = pfLead;
  PFCandidateRefVector pfTmp;
  selectedSignalPFChargedHadrCands_ =pfTmp;
  selectedSignalPFNeutrHadrCands_=pfTmp;
  selectedSignalPFGammaCands_ = pfTmp;

  selectedIsolationPFChargedHadrCands_ = pfTmp;
  selectedIsolationPFNeutrHadrCands_ = pfTmp;
  selectedIsolationPFGammaCands_ = pfTmp;

  selectedPFChargedHadrCands_ = pfTmp;
  selectedPFNeutrHadrCands_ = pfTmp;
  selectedPFGammaCands_ = pfTmp;

  maximumHcalTowerEnergy_ = NAN;
  leadPFChargedHadrCandsignedSipt_=NAN;
  leadTracksignedSipt_ =NAN;
  mass_ = NAN; 
  trackerMass_ = NAN;
  sumPtIsolation_=NAN;
  emOverHadronEnergy_=NAN;
  emIsolation_=NAN;
  numberOfEcalClusters_=std::numeric_limits<int>::quiet_NaN();
}
