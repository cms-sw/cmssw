#ifndef RecoMuon_TrackerSeedGenerator_H
#define RecoMuon_TrackerSeedGenerator_H

/** \class MuonSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: $
 *  $Revision: $
 *  \author Norbert Neumeister - Purdue University
 *  \porting author Chang Liu - Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/BasicTrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonSeedDetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonReconstructionEnumerators.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

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
    TrackerSeedGenerator(const MagneticField *field, edm::ParameterSet const& par);
    /// destructor
    virtual ~TrackerSeedGenerator();
    /// create seeds from muon trajectory
    BTSeedCollection findSeeds(const reco::Track& muon, const edm::Event& , const edm::EventSetup&) const;  

  private:

    void findLayerList(const TrajectoryStateOnSurface& traj);

    void primitiveSeeds(const reco::Track&, 
                        const TrajectoryStateOnSurface&);

    void consecutiveHitsSeeds(const reco::Track&, 
                              const TrajectoryStateOnSurface&, 
                              const TrackingRegion&);

    void createSeed(const MuonSeedDetLayer& outer,
                    const MuonSeedDetLayer& inner,
                    const TrackingRegion& regionOfInterest);

    void pixelSeeds(const reco::Track&, 
                    const TrajectoryStateOnSurface&, 
                    const TrackingRegion&,
                    float deltaEta, float deltaPhi);

    const MagneticField * theField;
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
};

#endif

