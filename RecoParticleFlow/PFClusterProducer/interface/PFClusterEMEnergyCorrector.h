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
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"

class PFClusterEMEnergyCorrector {
 public:
  PFClusterEMEnergyCorrector(const edm::ParameterSet& conf, edm::ConsumesCollector &&cc);
  PFClusterEMEnergyCorrector(const PFClusterEMEnergyCorrector&) = delete;
  PFClusterEMEnergyCorrector& operator=(const PFClusterEMEnergyCorrector&) = delete;

  void correctEnergies(const edm::Event &evt, const edm::EventSetup &es, const reco::PFCluster::EEtoPSAssociation &assoc, reco::PFClusterCollection& cs);

 private:    
  double maxPtForMVAEvaluation_;
       
  edm::EDGetTokenT<EBSrFlagCollection> ebSrFlagToken_; 
  edm::EDGetTokenT<EESrFlagCollection> eeSrFlagToken_; 

  //required for reading SR flags
  const EcalTrigTowerConstituentsMap * triggerTowerMap_;
  const EcalElectronicsMapping* elecMap_;

  edm::EDGetTokenT<EcalRecHitCollection> recHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEE_;  
  edm::EDGetTokenT<unsigned int> bunchSpacing_; 
  
  std::vector<std::string> condnames_mean_;
  std::vector<std::string> condnames_sigma_;  

  std::vector<std::string> condnames_mean_25ns_;
  std::vector<std::string> condnames_sigma_25ns_;  
  std::vector<std::string> condnames_mean_50ns_;
  std::vector<std::string> condnames_sigma_50ns_;  
    
  EcalTrigTowerDetId readOutUnitOf(const EBDetId& xtalId) const;
  EcalScDetId        readOutUnitOf(const EEDetId& xtalId) const;

  bool srfAwareCorrection_;
  bool applyCrackCorrections_;
  bool applyMVACorrections_;

  bool autoDetectBunchSpacing_;
  int bunchSpacingManual_;
        
  std::unique_ptr<PFEnergyCalibration> calibrator_;


};

#endif
