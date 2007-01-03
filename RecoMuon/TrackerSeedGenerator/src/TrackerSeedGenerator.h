#ifndef RecoMuon_TrackerSeedGenerator_H
#define RecoMuon_TrackerSeedGenerator_H

/** \class TrackerSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: 2006/11/10 17:27:21 $
 *  $Revision: 1.7 $
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

typedef std::vector<TrajectorySeed> BTSeedCollection;

class TrackerSeedGenerator {

  public:
    /// constructor
    TrackerSeedGenerator(const edm::ParameterSet& par, const MuonServiceProxy*);
    /// destructor
    virtual ~TrackerSeedGenerator();

    BTSeedCollection trackerSeeds(const Trajectory&, const RectangularEtaPhiTrackingRegion&);
    
    void setEvent(const edm::Event&);

  private:
    /// create seeds from muon trajectory
    void findSeeds(const Trajectory&, const RectangularEtaPhiTrackingRegion&); 

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

    const MuonServiceProxy *theService;
    GlobalPoint theVertexPos;
    GlobalError theVertexErr;
    CombinatorialRegionalSeedGeneratorFromPixel  combinatorialSeedGenerator;

    bool theMIMFlag;
    GlobalMuonMonitorInterface* dataMonitor;

};

#endif

