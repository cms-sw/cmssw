#ifndef FastElectronSeedGenerator_H
#define FastElectronSeedGenerator_H

/** \class FastElectronSeedGenerator

 * Class to generate the trajectory seed from two hits in
 *  the pixel detector which have been found compatible with
 *  an ECAL cluster.
 *
 * \author Patrick Janot
 *
 ************************************************************/
//UB added
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
//UB added
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

namespace edm {
  class EventSetup;
  class ParameterSet;
  class Event;
}

class TrackerGeometry;
class MagneticField;
class MagneticFieldMap;
class GeometricSearchTracker;
class TrackerInteractionGeometry;

class PropagatorWithMaterial;
class KFUpdator;
class FastPixelHitMatcher;
class TrackerRecHit;
class TrajectorySeed;
class TrackerTopology;

//UB changed
//class FastElectronSeedGenerator : public ElectronSeedGenerator
class FastElectronSeedGenerator
{
public:

  //RC
  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;


  enum mode{HLT, offline, unknown};  //to be used later

  FastElectronSeedGenerator(const edm::ParameterSet &pset,
			       double pTMin,
			       const edm::InputTag& beamSpot);

  ~FastElectronSeedGenerator();

  void setup(bool);

  void setupES(const edm::EventSetup& setup);

  //UB changed
/*  void run(edm::Event&,  */
/* 	   // const edm::Handle<reco::SuperClusterCollection>& clusters,   */
/* 	   const reco::SuperClusterRefVector &sclRefs, */
/* 	   const SiTrackerGSMatchedRecHit2DCollection*, */
/* 	   const edm::SimTrackContainer*, */
/* 	   reco::ElectronSeedCollection&); */
  void  run(edm::Event& e,
	    const reco::SuperClusterRefVector &sclRefs,
	    const SiTrackerGSMatchedRecHit2DCollection* theGSRecHits,
	    const edm::SimTrackContainer* theSimTracks,
	    TrajectorySeedCollection *seeds,
	    const TrackerTopology *tTopo,
	    reco::ElectronSeedCollection & out);

 private:

  void addASeedToThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
			     std::vector<TrackerRecHit>& theHits,
			     const TrajectorySeed& theTrackerSeed,
			     std::vector<reco::ElectronSeed>& result);

  bool prepareElTrackSeed(ConstRecHitPointer outerhit,
			  ConstRecHitPointer innerhit,
			  const GlobalPoint& vertexPos);

  bool dynamicphiroad_;

  float lowPtThreshold_;
  float highPtThreshold_;
  float sizeWindowENeg_;
  float phimin2_,phimax2_;
  float deltaPhi1Low_, deltaPhi1High_;
  float deltaPhi2_;
  bool searchInTIDTEC;

  double zmin1_, zmax1_;
  double pTMin2;
  math::XYZPoint BSPosition_;

  FastPixelHitMatcher *myGSPixelMatcher;

  //UB added
  TrajectorySeedCollection* theInitialSeedColl;
  bool fromTrackerSeeds_;

  const MagneticField* theMagField;
  const MagneticFieldMap* theMagneticFieldMap;
  const TrackerGeometry*  theTrackerGeometry;
  const GeometricSearchTracker* theGeomSearchTracker;
  const TrackerInteractionGeometry* theTrackerInteractionGeometry;

  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagator;

  const edm::EventSetup *theSetup;
  const edm::InputTag theBeamSpot;
  
  PRecHitContainer recHits_;
  PTrajectoryStateOnDet pts_;

};

#endif // FastElectronSeedGenerator_H


