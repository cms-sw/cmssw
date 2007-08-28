#ifndef RecoMuon_TrackerSeedGeneratorBC_H
#define RecoMuon_TrackerSeedGeneratorBC_H

/** \class TrackerSeedGeneratorBC
 *  Generate seed from muon trajectory.
 *
 *  $Date: 2007/04/18 17:21:10 $
 *  $Revision: 1.1 $
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"


class Trajectory;
class Propagator;
class MagneticField;
class TrackingRegion;
class RectangularEtaPhiTrackingRegion;
class MuonServiceProxy;
class MuonUpdatorAtVertex;
class Chi2MeasurementEstimator;

class GlobalMuonMonitorInterface;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class TrackerSeedGeneratorBC : public TrackerSeedGenerator {

 public:
  typedef std::vector<TrajectorySeed> BTSeedCollection;  
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;
  
  /// constructor
  TrackerSeedGeneratorBC(const edm::ParameterSet& par);
  virtual void init(const MuonServiceProxy*);

  TrackerSeedGeneratorBC(const edm::ParameterSet& par, const MuonServiceProxy*);
  /// destructor
  virtual ~TrackerSeedGeneratorBC();

  BTSeedCollection trackerSeeds(const TrackCand&, const TrackingRegion&);
    
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

  std::vector<TrajectorySeed> rsSeeds(const reco::Track&);

  //Propagator* thePropagator;
  //Propagator* theStepPropagator;
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
  std::string theRSPropagator;

  const MuonServiceProxy *theService;
  GlobalPoint theVertexPos;
  GlobalError theVertexErr;

  bool theMIMFlag;
  GlobalMuonMonitorInterface* dataMonitor;

  Chi2MeasurementEstimator * theRoadEstimator;
  edm::ParameterSet theConfig;


};

#endif

