#ifndef RecoMuon_GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define RecoMuon_GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: 2007/05/04 15:57:41 $
 *  $Revision: 1.56 $
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

namespace edm {class Event;}

class RectangularEtaPhiTrackingRegion;
class TrajectoryStateOnSurface;

class GlobalMuonTrackMatcher;
class TransientTrackingRecHitBuilder;
class MuonTransientTrackingRecHitBuilder;

class MuonDetLayerMeasurements;
class MuonTrackReFitter;
class MuonServiceProxy;
class Trajectory;
class TrackerSeedGenerator;
class TrajectoryCleaner;

class GlobalMuonMonitorInterface;

class TrackTransformer;

class GlobalMuonRSTrajectoryBuilder;
class TrajectoryFitter;
class MuonTrackingRegionBuilder;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder {

  public:

    typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;
    typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
    typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
    typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

    typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
    typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;

    typedef std::vector<Trajectory> TC;
    typedef TC::const_iterator TI;

  public:

    /// constructor with Parameter Set and MuonServiceProxy
    GlobalMuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*);
          
    /// destructor
    ~GlobalMuonTrajectoryBuilder();

    /// reconstruct trajectories from standalone and tracker only Tracks    
    MuonTrajectoryBuilder::CandidateContainer trajectories(const TrackCand&);

    /// dummy implementation, unused in this class
    MuonTrajectoryBuilder::TrajectoryContainer trajectories(const TrajectorySeed&) { return MuonTrajectoryBuilder::TrajectoryContainer(); }

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

  private:

    enum RefitDirection{inToOut,outToIn,undetermined};
    
    /// choose tracker tracks within region of interest
    std::vector<TrackCand> chooseRegionalTrackerTracks(const TrackCand&, 
                                                       const std::vector<TrackCand>&) const;

    /// define region of interest with tracker
    RectangularEtaPhiTrackingRegion defineRegionOfInterest(const reco::TrackRef&) const;

    /// build combined trajectory from sta Track and tracker RecHits
    MuonTrajectoryBuilder::CandidateContainer build(const TrackCand&, 
                                                    const std::vector<TrackCand>&) const;
  
    /// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
    void checkMuonHits(const reco::Track&, ConstRecHitContainer&, 
                       ConstRecHitContainer&, 
                       std::vector<int>&) const;
 
    /// select muon hits compatible with trajectory; check hits in chambers with showers
    ConstRecHitContainer selectMuonHits(const Trajectory&, 
                                        const std::vector<int>&) const;
 
    /// choose final trajectory
    const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&) const;

    /// choose final trajectory (new version)
    const Trajectory* chooseTrajectoryNew(const std::vector<Trajectory*>&) const;

    /// calculate chi2 probability (-ln(P))
    double trackProbability(const Trajectory&) const;    

    /// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

    /// build a tracker Trajectory from a seed
    TC makeTrajsFromSeeds(const std::vector<TrajectorySeed>&) const;

    /// make a TrackCand collection using tracker Track, Trajectory information
    std::vector<TrackCand> makeTkCandCollection(const TrackCand&) const;

    /// if TrackCand has only a TrackRef, attempt to add Trajectory*
    void addTraj(TrackCand&) const;

    RefitDirection checkRecHitsOrdering(const ConstRecHitContainer&) const;

    std::vector<Trajectory> refitTrajectory(const Trajectory*) const;

  private:

    GlobalPoint theVertexPos;
    GlobalError theVertexErr;
    GlobalMuonTrackMatcher* theTrackMatcher;
    MuonTrackReFitter* theRefitter;
    MuonDetLayerMeasurements* theLayerMeasurements;
    TrackerSeedGenerator* theTkSeedGenerator;
    TrajectoryCleaner* theTrajectoryCleaner;
    TrackTransformer* theTrackTransformer;
    MuonTrackingRegionBuilder* theRegionBuilder;
  
    int   theMuonHitsOption;
    NavigationDirection theDirection;
    float thePtCut;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;
    edm::InputTag theTkTrackLabel;
    std::string theCkfBuilderName;
    std::string trackerPropagatorName;
    std::string theKFFitterName;

    bool theTkTrajsAvailableFlag;
    bool theMakeTkSeedFlag;
    bool theRSFlag;
    bool theFirstEvent;
    bool theMIMFlag;

    GlobalMuonRSTrajectoryBuilder * theRSBuilder;

    edm::ESHandle<TrackerTrajectoryBuilder> theCkfBuilder;
    edm::Handle<reco::TrackCollection> allTrackerTracks;
    edm::ESHandle<TrajectoryFitter> theKFFitter;
    edm::Handle<TrajTrackAssociationCollection> tkAssoMap;

    const std::vector<Trajectory>* allTrackerTrajs;
 
    const edm::Event* theEvent;

    const MuonServiceProxy *theService;

    GlobalMuonMonitorInterface* dataMonitor;

    
};
#endif
