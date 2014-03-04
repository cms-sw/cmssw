#ifndef FastElectronSeedProducer_h
#define FastElectronSeedProducer_h

//
// Package:         FastSimulation/EgammaElectronAlgos
// Class:           FastElectronSeedProducer
//
// Description:     Calls FastElectronSeedGenerator
//                  to find TrackingSeeds.


class FastElectronSeedGenerator ;
class EgammaHcalIsolation ;

namespace edm {
  class EventSetup;
  class Event;
  class ParameterSet;
}

//#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
//#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class FastElectronSeedProducer : public edm::EDProducer
{

 public:

  explicit FastElectronSeedProducer(const edm::ParameterSet& conf);

  virtual ~FastElectronSeedProducer();

  virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

 private:

  void filterClusters(const edm::Handle<reco::SuperClusterCollection>& superClusters,
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
  FastElectronSeedGenerator * matcher_ ;

  // A few collections (seeds and hcal hits)
  const HBHERecHitCollection * hithbhe_ ;
  TrajectorySeedCollection * initialSeedColl_ ;

  // H/E filtering
  //HoECalculator calc_ ;
  edm::ESHandle<CaloGeometry> caloGeom_ ;
  unsigned long long caloGeomCacheId_ ;
  EgammaHcalIsolation * hcalIso_ ;
  //CaloDualConeSelector * doubleConeSel_ ;

  // maximum H/E where H is the Hcal energy inside the cone centered on the seed cluster eta-phi position
  double maxHOverE_ ;
  double hOverEConeSize_ ;
  double hOverEHBMinE_ ;
  double hOverEHFMinE_ ;
  double SCEtCut_ ;

  bool fromTrackerSeeds_ ;

};

#endif



