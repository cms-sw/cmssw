#ifndef RecoMuon_TrackerSeedGenerator_H
#define RecoMuon_TrackerSeedGenerator_H

/** \class TrackerSeedGenerator
 *
 *  Generate seed from muon trajectory.  Given a standalone muon
 *  TrackCand and a RectangularEtaPhiTrackingRegion around the muon,
 *  find a vector of TrajectorSeedthat are compatible with the muon.
 *
 *  $Date: 2007/03/07 17:41:02 $
 *  $Revision: 1.13 $
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
#include "DataFormats/Common/interface/Handle.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialRegionalSeedGeneratorFromPixel.h"

class Trajectory;
class Propagator;
class MagneticField;
class TrackingRegion;
class RectangularEtaPhiTrackingRegion;
class SeedGeneratorFromTrackingRegion;
class MuonServiceProxy;
class MuonUpdatorAtVertex;
class Chi2MeasurementEstimator;

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

  /// get a collection of TrajectorySeed
  BTSeedCollection trackerSeeds(const TrackCand&, const RectangularEtaPhiTrackingRegion&);
    
  /// initialize services
  void setEvent(const edm::Event&);

 private:
  /// create seeds from muon trajectory
  void findSeeds(const TrackCand&, const RectangularEtaPhiTrackingRegion&); 
  
  /// get list of compatible layers
  void findLayerList(const TrajectoryStateOnSurface& traj);
  
  /// make a PrimitiveMuonSeed
  void primitiveSeeds(const Trajectory&, 
		      const TrajectoryStateOnSurface&);

  /// make a seed from hits on consecutive layers
  void consecutiveHitsSeeds(const Trajectory&, 
			    const TrajectoryStateOnSurface&, 
			    const edm::EventSetup&,
			    const TrackingRegion&);

  /// make a seed from hits on two layers
  void createSeed(const MuonSeedDetLayer& outer,
		  const MuonSeedDetLayer& inner,
		  const edm::EventSetup&,
		  const TrackingRegion& regionOfInterest);

  /// make a seed from pixel pairs
  void pixelSeeds(const Trajectory&, 
		  const TrajectoryStateOnSurface&, 
		  const RectangularEtaPhiTrackingRegion&,
		  float deltaEta, float deltaPhi);

  /// make a seed based on the muon state at vertex
  std::vector<TrajectorySeed> rsSeeds(const reco::Track&);

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
  edm::ESHandle<GeometricSearchTracker> theSearchTracker;
  //const LayerMeasurements* theLayerMeasurements;
  //const MeasurementTracker* theMeasurementTracker;


  edm::Handle<SiPixelRecHitCollection> pixelHits;
  std::string hitProducer;
  std::string theOutPropagator;
  std::string theRSAnyPropagator;
  std::string theRSAlongPropagator;

  const MuonServiceProxy *theService;
  GlobalPoint theVertexPos;
  GlobalError theVertexErr;
  CombinatorialRegionalSeedGeneratorFromPixel  combinatorialSeedGenerator;

  bool theMIMFlag;
  GlobalMuonMonitorInterface* dataMonitor;

  Chi2MeasurementEstimator * theRoadEstimator;

};

#endif

