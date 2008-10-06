#ifndef CutBasedPhotonIDAlgo_H
#define CutBasedPhotonIDAlgo_H

#include "RecoEgamma/PhotonIdentification/interface/PhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CutBasedPhotonIDAlgo : PhotonIDAlgo {

public:

  CutBasedPhotonIDAlgo(){};

  virtual ~CutBasedPhotonIDAlgo(){};

  void setup(const edm::ParameterSet& conf);
  reco::PhotonID calculate(const reco::Photon*, const edm::Event&, const edm::EventSetup& es);
  void decideEB(reco::PhotonID &phID, const reco::Photon* pho);
  void decideEE(reco::PhotonID &phID, const reco::Photon* pho);
 private:
  
  //Which cuts to do?
  bool dophotonBCIsolationCut_;
  bool dophotonEcalRecHitIsolationCut_;
  bool dophotonHcalTowerIsolationCut_;
  bool dophotonHCTrkIsolationCut_;
  bool dophotonSCTrkIsolationCut_;
  bool dophotonHCNTrkCut_;
  bool dophotonSCNTrkCut_;
  bool dorequireFiducial_;
  bool dophotonHadOverEMCut_;
  bool dophotonsigmaeeCut_;
  bool dophotonR9Cut_;

  //Actual cut values
  double looseEMEcalRecHitIsolationCutEB_;
  double looseEMHcalTowerIsolationCutEB_;
  double looseEMHollowConeTrkIsolationCutEB_;
  double looseEMSolidConeTrkIsolationCutEB_;
  int looseEMSolidConeNTrkCutEB_;
  int looseEMHollowConeNTrkCutEB_;
  double looseEMEtaWidthCutEB_;
  double looseEMHadOverEMCutEB_;
  double looseEMR9CutEB_;

  double loosephotonEcalRecHitIsolationCutEB_;
  double loosephotonHcalTowerIsolationCutEB_;
  double loosephotonHollowConeTrkIsolationCutEB_;
  double loosephotonSolidConeTrkIsolationCutEB_;
  int loosephotonSolidConeNTrkCutEB_;
  int loosephotonHollowConeNTrkCutEB_;
  double loosephotonEtaWidthCutEB_;
  double loosephotonHadOverEMCutEB_;
  double loosephotonR9CutEB_;

  double tightphotonEcalRecHitIsolationCutEB_;
  double tightphotonHcalTowerIsolationCutEB_;
  double tightphotonHollowConeTrkIsolationCutEB_;
  double tightphotonSolidConeTrkIsolationCutEB_;
  int tightphotonSolidConeNTrkCutEB_;
  int tightphotonHollowConeNTrkCutEB_;
  double tightphotonEtaWidthCutEB_;
  double tightphotonHadOverEMCutEB_;
  double tightphotonR9CutEB_;

  double looseEMEcalRecHitIsolationCutEE_;
  double looseEMHcalTowerIsolationCutEE_;
  double looseEMHollowConeTrkIsolationCutEE_;
  double looseEMSolidConeTrkIsolationCutEE_;
  int looseEMSolidConeNTrkCutEE_;
  int looseEMHollowConeNTrkCutEE_;
  double looseEMEtaWidthCutEE_;
  double looseEMHadOverEMCutEE_;
  double looseEMR9CutEE_;

  double loosephotonEcalRecHitIsolationCutEE_;
  double loosephotonHcalTowerIsolationCutEE_;
  double loosephotonHollowConeTrkIsolationCutEE_;
  double loosephotonSolidConeTrkIsolationCutEE_;
  int loosephotonSolidConeNTrkCutEE_;
  int loosephotonHollowConeNTrkCutEE_;
  double loosephotonEtaWidthCutEE_;
  double loosephotonHadOverEMCutEE_;
  double loosephotonR9CutEE_;

  double tightphotonEcalRecHitIsolationCutEE_;
  double tightphotonHcalTowerIsolationCutEE_;
  double tightphotonHollowConeTrkIsolationCutEE_;
  double tightphotonSolidConeTrkIsolationCutEE_;
  int tightphotonSolidConeNTrkCutEE_;
  int tightphotonHollowConeNTrkCutEE_;
  double tightphotonEtaWidthCutEE_;
  double tightphotonHadOverEMCutEE_;
  double tightphotonR9CutEE_;

  //Isolation parameters
  double photonEcalRecHitConeInnerRadius_;
  double photonEcalRecHitConeOuterRadius_;
  double photonEcalRecHitEtaSlice_;
  double photonEcalRecHitThreshE_;
  double photonEcalRecHitThreshEt_;
  double photonHcalTowerConeInnerRadius_;
  double photonHcalTowerConeOuterRadius_;
  double photonHcalTowerThreshE_;
  double trackConeOuterRadius_;
  double trackConeInnerRadius_;
  double isolationtrackThreshold_;
 
};

#endif // CutBasedPhotonIDAlgo_H
