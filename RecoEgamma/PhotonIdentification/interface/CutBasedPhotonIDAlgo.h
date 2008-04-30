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
  void decide(reco::PhotonID &phID, const reco::Photon* pho);
 private:
  
  //Which cuts to do?
  bool dophotonBCIsolationCut_;
  bool dophotonEcalRecHitIsolationCut_;
  bool dophotonHcalRecHitIsolationCut_;
  bool dophotonHCTrkIsolationCut_;
  bool dophotonSCTrkIsolationCut_;
  bool dophotonHCNTrkCut_;
  bool dophotonSCNTrkCut_;
  bool dorequireNotElectron_;
  bool dorequireFiducial_;
  bool dophotonHadOverEMCut_;
  bool dophotonsigmaeeCut_;
  bool dophotonR9Cut_;

  //Actual cut values
  double looseEMEcalRecHitIsolationCut_;
  double looseEMHcalRecHitIsolationCut_;
  double looseEMHollowConeTrkIsolationCut_;
  double looseEMSolidConeTrkIsolationCut_;
  int looseEMSolidConeNTrkCut_;
  int looseEMHollowConeNTrkCut_;
  double looseEMEtaWidthCut_;
  double looseEMHadOverEMCut_;
  double looseEMR9Cut_;

  double loosephotonEcalRecHitIsolationCut_;
  double loosephotonHcalRecHitIsolationCut_;
  double loosephotonHollowConeTrkIsolationCut_;
  double loosephotonSolidConeTrkIsolationCut_;
  int loosephotonSolidConeNTrkCut_;
  int loosephotonHollowConeNTrkCut_;
  double loosephotonEtaWidthCut_;
  double loosephotonHadOverEMCut_;
  double loosephotonR9Cut_;

  double tightphotonEcalRecHitIsolationCut_;
  double tightphotonHcalRecHitIsolationCut_;
  double tightphotonHollowConeTrkIsolationCut_;
  double tightphotonSolidConeTrkIsolationCut_;
  int tightphotonSolidConeNTrkCut_;
  int tightphotonHollowConeNTrkCut_;
  double tightphotonEtaWidthCut_;
  double tightphotonHadOverEMCut_;
  double tightphotonR9Cut_;

  //Isolation parameters
  double photonEcalRecHitConeInnerRadius_;
  double photonEcalRecHitConeOuterRadius_;
  double photonEcalRecHitThresh_;
  double photonHcalRecHitConeInnerRadius_;
  double photonHcalRecHitConeOuterRadius_;
  double photonHcalRecHitThresh_;
  double trackConeOuterRadius_;
  double trackConeInnerRadius_;
  double isolationtrackThreshold_;
 
};

#endif // CutBasedPhotonIDAlgo_H
