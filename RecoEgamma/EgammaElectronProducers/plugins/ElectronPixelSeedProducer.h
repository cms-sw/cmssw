#ifndef ElectronPixelSeedProducer_h
#define ElectronPixelSeedProducer_h
  
//
// Package:         RecoEgamma/ElectronTrackSeedProducers
// Class:           ElectronPixelSeedProducer
// 
// Description:     Calls ElectronPixelSeedGenerator
//                  to find TrackingSeeds.
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class  ElectronSeedGenerator;

class ElectronPixelSeedProducer : public edm::EDProducer
{
 public:
  
  explicit ElectronPixelSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~ElectronPixelSeedProducer();
  
  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:

  void filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters,HBHERecHitMetaCollection* mhbhe, reco::SuperClusterRefVector &sclRefs);

  edm::InputTag superClusters_[2];
  edm::InputTag hcalRecHits_;
  
  const edm::ParameterSet conf_;
  ElectronSeedGenerator *matcher_;
  std::string algo_;
 
  const HBHERecHitCollection* hithbhe_;

  //for the filter
  edm::ESHandle<CaloGeometry>       theCaloGeom;
  HoECalculator calc_;

  // maximum H/E where H is the Hcal energy in tower behind the seed cluster eta-phi position 
  double maxHOverE_; 
  double SCEtCut_;

  unsigned long long cacheID_;
};
  
#endif
 


