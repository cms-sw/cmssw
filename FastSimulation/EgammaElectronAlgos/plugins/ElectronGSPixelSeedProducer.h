#ifndef ElectronGSPixelSeedProducer_h
#define ElectronGSPixelSeedProducer_h
  
//
// Package:         FastSimulation/EgammaElectronAlgos
// Class:           ElectronGSPixelSeedProducer
// 
// Description:     Calls ElectronGSPixelSeedGenerator
//                  to find TrackingSeeds.
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
  
class  ElectronGSPixelSeedGenerator;

namespace edm { 
  class EventSetup;
  class Event;
  class ParameterSet;
}

class ElectronSeedGenerator;
 
class ElectronGSPixelSeedProducer : public edm::EDProducer
{
 public:
  
  explicit ElectronGSPixelSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~ElectronGSPixelSeedProducer();
  
  virtual void beginJob(edm::EventSetup const& iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:

  void filterClusters(const reco::SuperClusterCollection* superClusters,
		      HBHERecHitMetaCollection* mhbhe, 
		      reco::SuperClusterRefVector& sclRefs);


 private:
  edm::InputTag clusters_[2];
  edm::InputTag simTracks_;
  edm::InputTag trackerHits_;
  edm::InputTag hcalRecHits_;
  std::string algo;

  // Pixel Seed generator
  ElectronSeedGenerator *matcher_;

  // H/E filtering
  HoECalculator calc_;
 
  // maximum H/E where H is the Hcal energy inside the cone centered on the seed cluster eta-phi position 
  double maxHOverE_; 
  double SCEtCut_;

  };
  
#endif
 


