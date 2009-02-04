#ifndef FastElectronSeedProducer_h
#define FastElectronSeedProducer_h
  
//
// Package:         FastSimulation/EgammaElectronAlgos
// Class:           FastElectronSeedProducer
// 
// Description:     Calls FastElectronSeedGenerator
//                  to find TrackingSeeds.
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
  
class FastElectronSeedGenerator;

namespace edm { 
  class EventSetup;
  class Event;
  class ParameterSet;
}

 
class FastElectronSeedProducer : public edm::EDProducer
{

 public:
  
  explicit FastElectronSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~FastElectronSeedProducer();
  
  virtual void beginRun(edm::Run & run, const edm::EventSetup & es);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:

  void filterClusters(const edm::Handle<reco::SuperClusterCollection>& superClusters,
		      HBHERecHitMetaCollection* mhbhe, 
		      reco::SuperClusterRefVector& sclRefs);

  /*
  //UB added
  void filterSeeds(edm::Event& e, const edm::EventSetup& setup, reco::SuperClusterRefVector &sclRefs);
  */

 private:
  // Input Tags
  edm::InputTag clusters_[2];
  edm::InputTag simTracks_;
  edm::InputTag trackerHits_;
  edm::InputTag hcalRecHits_;
  edm::InputTag initialSeeds_;

  // Pixel Seed generator
  FastElectronSeedGenerator *matcher_;

  // A few collections (seeds and hcal hits)
  const HBHERecHitCollection* hithbhe_;
  TrajectorySeedCollection *theInitialSeedColl;

  // H/E filtering
  HoECalculator calc_;
 
  // maximum H/E where H is the Hcal energy inside the cone centered on the seed cluster eta-phi position 
  double maxHOverE_; 
  double SCEtCut_;

  bool fromTrackerSeeds_;

};
  
#endif
 


