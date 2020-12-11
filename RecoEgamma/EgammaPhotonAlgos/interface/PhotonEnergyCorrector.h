#ifndef PhotonEnergyCorrector_H
#define PhotonEnergyCorrector_H
/** \class PhotonEnergyCorrector
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrector.h"
#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/EnergyUncertaintyPhotonSpecific.h"
#include <iostream>

class PhotonEnergyCorrector {
public:
  PhotonEnergyCorrector(const edm::ParameterSet& config, edm::ConsumesCollector&& iC);

  std::unique_ptr<ModifyObjectValueBase>& gedRegression() { return gedRegression_; }

  void init(const edm::EventSetup& theEventSetup);
  void calculate(
      edm::Event& evt, reco::Photon&, int subdet, const reco::VertexCollection& vtxcol, const edm::EventSetup& iSetup);
  double applyCrackCorrection(const reco::SuperCluster& cl, EcalClusterFunctionBaseClass* crackCorrectionFunction);

private:
  bool weightsfromDB_;
  std::string w_file_;
  std::string w_db_;
  std::string candidateP4type_;
  std::unique_ptr<EGEnergyCorrector> regressionCorrector_;
  std::unique_ptr<EcalClusterFunctionBaseClass> scEnergyFunction_;
  std::unique_ptr<EcalClusterFunctionBaseClass> scCrackEnergyFunction_;
  std::unique_ptr<EcalClusterFunctionBaseClass> scEnergyErrorFunction_;
  std::unique_ptr<EcalClusterFunctionBaseClass> photonEcalEnergyCorrFunction_;
  std::unique_ptr<ModifyObjectValueBase> gedRegression_;
  double minR9Barrel_;
  double minR9Endcap_;
  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHitsToken_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHitsToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  std::unique_ptr<EnergyUncertaintyPhotonSpecific> photonUncertaintyCalculator_;
};

#endif
