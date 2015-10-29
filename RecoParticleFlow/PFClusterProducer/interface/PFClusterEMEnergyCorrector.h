#ifndef __PFClusterEMEnergyCorrector_H__
#define __PFClusterEMEnergyCorrector_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class PFClusterEMEnergyCorrector {
 public:
  PFClusterEMEnergyCorrector(const edm::ParameterSet& conf, edm::ConsumesCollector &&cc);
  PFClusterEMEnergyCorrector(const PFClusterEMEnergyCorrector&) = delete;
  PFClusterEMEnergyCorrector& operator=(const PFClusterEMEnergyCorrector&) = delete;

  void correctEnergies(const edm::Event &evt, const edm::EventSetup &es, const reco::PFCluster::EEtoPSAssociation &assoc, reco::PFClusterCollection& cs);

 private:    
  bool _applyCrackCorrections;
  bool _applyMVACorrections;
  double _maxPtForMVAEvaluation;
   
  bool autoDetectBunchSpacing_;
  int bunchSpacingManual_;
  
  edm::EDGetTokenT<unsigned int> bunchSpacing_; 
  
  edm::EDGetTokenT<EcalRecHitCollection> _recHitsEB;
  edm::EDGetTokenT<EcalRecHitCollection> _recHitsEE;  
  
  std::vector<std::string> _condnames_mean_50ns;
  std::vector<std::string> _condnames_sigma_50ns;
  std::vector<std::string> _condnames_mean_25ns;
  std::vector<std::string> _condnames_sigma_25ns;  
  
   std::unique_ptr<PFEnergyCalibration> _calibrator;
  
};

#endif
