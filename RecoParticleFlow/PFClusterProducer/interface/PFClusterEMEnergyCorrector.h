#ifndef __PFClusterEMEnergyCorrector_H__
#define __PFClusterEMEnergyCorrector_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EBSrFlag.h"
#include "DataFormats/EcalDigi/interface/EESrFlag.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalReadoutTools.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"

class PFClusterEMEnergyCorrector {
public:
  PFClusterEMEnergyCorrector(const edm::ParameterSet& conf, edm::ConsumesCollector&& cc);
  PFClusterEMEnergyCorrector(const PFClusterEMEnergyCorrector&) = delete;
  PFClusterEMEnergyCorrector& operator=(const PFClusterEMEnergyCorrector&) = delete;

  void correctEnergies(const edm::Event& evt,
                       const edm::EventSetup& es,
                       const reco::PFCluster::EEtoPSAssociation& assoc,
                       reco::PFClusterCollection& cs);

private:
  double maxPtForMVAEvaluation_;

  edm::EDGetTokenT<EBSrFlagCollection> ebSrFlagToken_;
  edm::EDGetTokenT<EESrFlagCollection> eeSrFlagToken_;

  //required for reading SR flags
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEE_;
  edm::EDGetTokenT<unsigned int> bunchSpacing_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  std::vector<std::string> condnames_mean_;
  std::vector<std::string> condnames_sigma_;

  std::vector<std::string> condnames_mean_25ns_;
  std::vector<std::string> condnames_sigma_25ns_;
  std::vector<std::string> condnames_mean_50ns_;
  std::vector<std::string> condnames_sigma_50ns_;

  bool srfAwareCorrection_;
  bool applyCrackCorrections_;
  bool applyMVACorrections_;
  bool setEnergyUncertainty_;

  bool autoDetectBunchSpacing_;
  int bunchSpacingManual_;

  std::unique_ptr<PFEnergyCalibration> calibrator_;
  void getAssociatedPSEnergy(const size_t clusIdx,
                             const reco::PFCluster::EEtoPSAssociation& assoc,
                             float& e1,
                             float& e2);

  double meanlimlowEB_;
  double meanlimhighEB_;
  double meanoffsetEB_;
  double meanscaleEB_;

  double meanlimlowEE_;
  double meanlimhighEE_;
  double meanoffsetEE_;
  double meanscaleEE_;

  double sigmalimlowEB_;
  double sigmalimhighEB_;
  double sigmaoffsetEB_;
  double sigmascaleEB_;

  double sigmalimlowEE_;
  double sigmalimhighEE_;
  double sigmaoffsetEE_;
  double sigmascaleEE_;
};

#endif
