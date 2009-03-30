#ifndef CutBasedPhotonIDAlgo_H
#define CutBasedPhotonIDAlgo_H

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class CutBasedPhotonIDAlgo  {

public:

  CutBasedPhotonIDAlgo(){};

  virtual ~CutBasedPhotonIDAlgo(){};

  void setup(const edm::ParameterSet& conf);
  void decideEB(const reco::Photon* pho, 
		bool &LoosePhoton, 
		bool &TightPhoton 
		);
  void decideEE(const reco::Photon* pho, 
		bool &LoosePhoton,
		bool &TightPhoton);
 private:
  
  //Which cuts to do?
  bool dophotonEcalRecHitIsolationCut_;
  bool dophotonEcalIsoRelativeCut_;
  bool dophotonHcalTowerIsolationCut_;
  bool dophotonHCTrkIsolationCut_;
  bool dophotonSCTrkIsolationCut_;
  bool dophotonHCNTrkCut_;
  bool dophotonSCNTrkCut_;
  bool dorequireFiducial_;
  bool dophotonHadOverEMCut_;
  bool dophotonsigmaeeCut_;
  bool dophotonR9Cut_;

  double loosephotonEcalRecHitIsolationCutEB_;
  double loosephotonEcalIsoRelativeCutEB_;
  double loosephotonHcalTowerIsolationCutEB_;
  double loosephotonHollowConeTrkIsolationCutEB_;
  double loosephotonSolidConeTrkIsolationCutEB_;
  int loosephotonSolidConeNTrkCutEB_;
  int loosephotonHollowConeNTrkCutEB_;
  double loosephotonEtaWidthCutEB_;
  double loosephotonHadOverEMCutEB_;
  double loosephotonR9CutEB_;

  double tightphotonEcalRecHitIsolationCutEB_;
  double tightphotonEcalIsoRelativeCutEB_;
  double tightphotonHcalTowerIsolationCutEB_;
  double tightphotonHollowConeTrkIsolationCutEB_;
  double tightphotonSolidConeTrkIsolationCutEB_;
  int tightphotonSolidConeNTrkCutEB_;
  int tightphotonHollowConeNTrkCutEB_;
  double tightphotonEtaWidthCutEB_;
  double tightphotonHadOverEMCutEB_;
  double tightphotonR9CutEB_;

  double loosephotonEcalRecHitIsolationCutEE_;
  double loosephotonEcalIsoRelativeCutEE_;
  double loosephotonHcalTowerIsolationCutEE_;
  double loosephotonHollowConeTrkIsolationCutEE_;
  double loosephotonSolidConeTrkIsolationCutEE_;
  int loosephotonSolidConeNTrkCutEE_;
  int loosephotonHollowConeNTrkCutEE_;
  double loosephotonEtaWidthCutEE_;
  double loosephotonHadOverEMCutEE_;
  double loosephotonR9CutEE_;

  double tightphotonEcalRecHitIsolationCutEE_;
  double tightphotonEcalIsoRelativeCutEE_;
  double tightphotonHcalTowerIsolationCutEE_;
  double tightphotonHollowConeTrkIsolationCutEE_;
  double tightphotonSolidConeTrkIsolationCutEE_;
  int tightphotonSolidConeNTrkCutEE_;
  int tightphotonHollowConeNTrkCutEE_;
  double tightphotonEtaWidthCutEE_;
  double tightphotonHadOverEMCutEE_;
  double tightphotonR9CutEE_;

 
};

#endif // CutBasedPhotonIDAlgo_H
