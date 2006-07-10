#ifndef RecoMuon_TrackerSeedGenerator_H
#define RecoMuon_TrackerSeedGenerator_H

/** \class TrackerSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: 2006/06/12 02:17:55 $
 *  $Revision: 1.4 $
 *  \author Norbert Neumeister - Purdue University
 *  \porting author Chang Liu - Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/BasicTrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonSeedDetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonReconstructionEnumerators.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"


class Trajectory;
class Propagator;
class MagneticField;
class TrackingRegion;
class SeedGeneratorFromTrackingRegion;

namespace edm {class ParameterSet; class Event; class EventSetup;}

typedef std::vector<BasicTrajectorySeed*> BTSeedCollection;

class TrackerSeedGenerator {

  public:
    /// constructor
    TrackerSeedGenerator(edm::ParameterSet const& par, const edm::EventSetup&);
    /// destructor
    virtual ~TrackerSeedGenerator();

    BTSeedCollection trackerSeeds(const Trajectory&, const edm::Event& , const edm::EventSetup&);
    
  private:
    /// create seeds from muon trajectory
    void findSeeds(const Trajectory& muon, const edm::Event& , const edm::EventSetup&);  

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
                    const edm::EventSetup&,
                    const TrackingRegion&,
                    float deltaEta, float deltaPhi);

    Propagator* thePropagator;
    Propagator* theStepPropagator;
    SeedGeneratorFromTrackingRegion* theSeedGenerator;
    GlobalPoint theVertexPos;
    GlobalError theVertexErr;
    ReconstructionDirection theDirection;
    int theOption;
    bool theUseVertex;
    int theMaxSeeds;
    int theMaxLayers;
    float theErrorRescale;
    std::vector<MuonSeedDetLayer> theLayerList;
    BTSeedCollection theSeeds;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    edm::ESHandle<MagneticField> theField;
    edm::ESHandle<GeometricSearchTracker> theGeoTracker;
    const LayerMeasurements* theLayerMeasurements;
    const MeasurementTracker* theMeasurementTracker;

};

#endif

