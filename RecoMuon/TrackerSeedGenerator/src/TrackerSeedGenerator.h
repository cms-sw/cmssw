#ifndef RecoMuon_TrackerSeedGenerator_H
#define RecoMuon_TrackerSeedGenerator_H

/** \class TrackerSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: 2007/02/14 06:12:39 $
 *  $Revision: 1.10 $
 *  \author Norbert Neumeister - Purdue University
 *  \porting author Chang Liu - Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/BasicTrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonSeedDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialRegionalSeedGeneratorFromPixel.h"

class Trajectory;
class Propagator;
class MagneticField;
class TrackingRegion;
class RectangularEtaPhiTrackingRegion;
class SeedGeneratorFromTrackingRegion;
class MuonServiceProxy;
class MuonUpdatorAtVertex;

class GlobalMuonMonitorInterface;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class TrackerSeedGenerator {

 public:
  typedef std::vector<TrajectorySeed> BTSeedCollection;  
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;
  
  /// constructor
  TrackerSeedGenerator(const edm::ParameterSet& par, const MuonServiceProxy*);
  /// destructor
  virtual ~TrackerSeedGenerator();

  BTSeedCollection trackerSeeds(const TrackCand&, const RectangularEtaPhiTrackingRegion&);
    
  void setEvent(const edm::Event&);

 private:
  /// create seeds from muon trajectory
  void findSeeds(const TrackCand&, const RectangularEtaPhiTrackingRegion&); 

  void findLayerList(const TrajectoryStateOnSurface& traj);

  void primitiveSeeds(const Trajectory&, 
		      const TrajectoryStateOnSurface&);

  void consecutiveHitsSeeds(const Trajectory&, 
			    const TrajectoryStateOnSurface&, 
			    const edm::EventSetup&,
			    const TrackingRegion&);

  void createSeed(const MuonSeedDetLayer& outer,
		  const MuonSeedDetLayer& inner,
		  const edm::EventSetup&,
		  const TrackingRegion& regionOfInterest);

  void pixelSeeds(const Trajectory&, 
		  const TrajectoryStateOnSurface&, 
		  const RectangularEtaPhiTrackingRegion&,
		  float deltaEta, float deltaPhi);

  //Propagator* thePropagator;
  //Propagator* theStepPropagator;
  //SeedGeneratorFromTrackingRegion* theSeedGenerator;
  MuonUpdatorAtVertex* theUpdator;

  //ReconstructionDirection theDirection;
  int theOption;
  //bool theUseVertex;
  int theMaxSeeds;
  //int theMaxLayers;
  float theErrorRescale;
  std::vector<MuonSeedDetLayer> theLayerList;
  BTSeedCollection theSeeds;
  //edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  //edm::ESHandle<MagneticField> theField;
  //GeometricSearchTracker* theGeoTracker;
  //const LayerMeasurements* theLayerMeasurements;
  //const MeasurementTracker* theMeasurementTracker;


  edm::Handle<SiPixelRecHitCollection> pixelHits;
  std::string hitProducer;
  std::string theOutPropagator;

  const MuonServiceProxy *theService;
  GlobalPoint theVertexPos;
  GlobalError theVertexErr;
  CombinatorialRegionalSeedGeneratorFromPixel  combinatorialSeedGenerator;

  bool theMIMFlag;
  GlobalMuonMonitorInterface* dataMonitor;

};

#endif

