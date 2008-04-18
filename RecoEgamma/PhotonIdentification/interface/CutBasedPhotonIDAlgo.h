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

  //Actual cut values
  double photonBasicClusterIsolationCut_;
  double photonEcalRecHitIsolationCut_;
  double photonHcalRecHitIsolationCut_;
  double photonHollowConeTrkIsolationCut_;
  double photonSolidConeTrkIsolationCut_;
  int photonSolidConeNTrkCut_;
  int photonHollowConeNTrkCut_;
  double photonEtaWidthCut_;
  double photonHadOverEMCut_;


  //Isolation parameters
  double photonBasicClusterConeOuterRadius_;
  double photonBasicClusterConeInnerRadius_;
  double photonEcalRecHitConeInnerRadius_;
  double photonEcalRecHitConeOuterRadius_;
  double photonEcalRecHitThresh_;
  double photonHcalRecHitConeInnerRadius_;
  double photonHcalRecHitConeOuterRadius_;
  double photonHcalRecHitThresh_;
  double isolationbasicclusterThreshold_;
  double trackConeOuterRadius_;
  double trackConeInnerRadius_;
  double isolationtrackThreshold_;
 
};

#endif // CutBasedPhotonIDAlgo_H
