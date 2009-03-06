#ifndef ElectronSeedProducer_h
#define ElectronSeedProducer_h

//
// Package:         RecoEgamma/ElectronTrackSeedProducers
// Class:           ElectronSeedProducer
//
// Description:     Calls ElectronSeedGenerator
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
class SeedFilter;
///class TrajectorySeedCollection;

class ElectronSeedProducer : public edm::EDProducer
{
 public:

  explicit ElectronSeedProducer(const edm::ParameterSet& conf);

  virtual ~ElectronSeedProducer();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  void filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters,
      HBHERecHitMetaCollection*mhbhe, reco::SuperClusterRefVector &sclRefs);
  void filterSeeds(edm::Event& e, const edm::EventSetup& setup, reco::SuperClusterRefVector &sclRefs);

  edm::InputTag superClusters_[2];
  edm::InputTag hcalRecHits_;
  edm::InputTag initialSeeds_;

  const edm::ParameterSet conf_;
  ElectronSeedGenerator *matcher_;
  SeedFilter * seedFilter_;

  TrajectorySeedCollection *theInitialSeedColl;

  //for the filter
  edm::ESHandle<CaloGeometry>       theCaloGeom;
  HoECalculator calc_;

  // parameters for H/E
  double maxHOverE_;

  // super cluster Et cut
  double SCEtCut_;

  unsigned long long cacheID_;

  bool fromTrackerSeeds_;
  bool prefilteredSeeds_;

};

#endif



