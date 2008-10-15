#ifndef CutBasedPhotonIDAlgo_H
#define CutBasedPhotonIDAlgo_H

#include "RecoEgamma/PhotonIdentification/interface/PhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct CutBasedPhotonQuantities
{
  //Did this pass the cuts in the configuration?
  bool isLooseEM_;
  bool isLoosePhoton_;
  bool isTightPhoton_;
  
  //These are analysis quantities calculated in the PhotonIDAlgo class
  //EcalRecHit isolation
  float isolationEcalRecHitA_;
  //HcalRecHit isolation
  float isolationHcalTowerA_;
  //Sum of track pT in a cone of dR
  float isolationSolidTrkConeA_;
  //Sum of track pT in a hollow cone of outer radius, inner radius
  float isolationHollowTrkConeA_;
  //Number of tracks in a cone of dR
  int nTrkSolidConeA_;
  //Number of tracks in a hollow cone of outer radius, inner radius
  int nTrkHollowConeA_;

  //EcalRecHit isolation
  float isolationEcalRecHitB_;
  //HcalRecHit isolation
  float isolationHcalTowerB_;
  //Sum of track pT in a cone of dR
  float isolationSolidTrkConeB_;
  //Sum of track pT in a hollow cone of outer radius, inner radius
  float isolationHollowTrkConeB_;
  //Number of tracks in a cone of dR
  int nTrkSolidConeB_;
  //Number of tracks in a hollow cone of outer radius, inner radius
  int nTrkHollowConeB_;
  //Highest energy 2x5 array
  float e2x5_;
  //Highest energy 1x5 array
  float e1x5_;
  //log weighted width in ieta
  float sigmaIetaIeta_;
  float r9_;
  //Fiducial flags
  bool isEBPho_;//Photon is in EB
  bool isEEPho_;//Photon is in EE
  bool isEBGap_;//Photon is in supermodule/supercrystal gap in EB
  bool isEEGap_;//Photon is in crystal gap in EE
  bool isEBEEGap_;//Photon is in border between EB and EE.

};

class CutBasedPhotonIDAlgo : PhotonIDAlgo {

public:

  CutBasedPhotonIDAlgo(){};

  virtual ~CutBasedPhotonIDAlgo(){};

  void setup(const edm::ParameterSet& conf);
  void calculate(const reco::Photon*, const edm::Event&, const edm::EventSetup& es, CutBasedPhotonQuantities &phoid);
  void decideEB(CutBasedPhotonQuantities &phID, const reco::Photon* pho);
  void decideEE(CutBasedPhotonQuantities &phID, const reco::Photon* pho);
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
  double photonEcalRecHitConeInnerRadiusA_;
  double photonEcalRecHitConeOuterRadiusA_;
  double photonEcalRecHitEtaSliceA_;
  double photonEcalRecHitThreshEA_;
  double photonEcalRecHitThreshEtA_;
  double photonHcalTowerConeInnerRadiusA_;
  double photonHcalTowerConeOuterRadiusA_;
  double photonHcalTowerThreshEA_;
  double trackConeOuterRadiusA_;
  double trackConeInnerRadiusA_;
  double isolationtrackThresholdA_;

  double photonEcalRecHitConeInnerRadiusB_;
  double photonEcalRecHitConeOuterRadiusB_;
  double photonEcalRecHitEtaSliceB_;
  double photonEcalRecHitThreshEB_;
  double photonEcalRecHitThreshEtB_;
  double photonHcalTowerConeInnerRadiusB_;
  double photonHcalTowerConeOuterRadiusB_;
  double photonHcalTowerThreshEB_;
  double trackConeOuterRadiusB_;
  double trackConeInnerRadiusB_;
  double isolationtrackThresholdB_;
 
};

#endif // CutBasedPhotonIDAlgo_H
