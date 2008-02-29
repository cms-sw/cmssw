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

  std::string instanceName_[2];
  std::string label_[2];
  std::string hbheLabel_;
  std::string hbheInstanceName_;

  const edm::ParameterSet conf_;
  ElectronSeedGenerator *matcher_;
 
  const CaloSubdetectorGeometry *subDetGeometry_; 
  const HBHERecHitCollection* hithbhe_;

  // cuts for the filter
  // cone size for H/E
  double hOverEConeSize_; 
  // maximum H/E where H is the Hcal energy inside the cone centered on the seed cluster eta-phi position 
  double maxHOverE_; 
  double SCEtCut_;
};
  
#endif
 


