#ifndef GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: 2006/08/03 13:17:57 $
 *  $Revision: 1.23 $
 *  \author Norbert Neumeister - Purdue University
 *  \author Chang Liu - Purdue University
 */

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonReconstructionEnumerators.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

class RectangularEtaPhiTrackingRegion;
class TrajectoryStateOnSurface;
class MuonUpdatorAtVertex;
class MagneticField;
class GlobalMuonTrackMatcher;
class TransientTrackingRecHit;
class MuonTransientTrackingRecHit;
class TransientTrackingRecHitBuilder;
class GlobalTrackingGeometry;
class MuonDetLayerGeometry;
class GlobalMuonReFitter;
class Propagator;
class TrajectoryFitter;
class MuonDetLayerMeasurements;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder {

  public:

    typedef TransientTrackingRecHit::RecHitContainer   RecHitContainer;
    typedef TransientTrackingRecHit::ConstRecHitContainer  ConstRecHitContainer;
    typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
    typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

    typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
    typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;

    typedef std::vector<Trajectory> TC;
    typedef TC::const_iterator TI;

  public:

    /// constructor with Parameter Set
    GlobalMuonTrajectoryBuilder(const edm::ParameterSet&);
          
    /// destructor
    ~GlobalMuonTrajectoryBuilder();

    /// reconstruct trajectories from standalone and tracker only Tracks
    MuonTrajectoryBuilder::CandidateContainer trajectories(const reco::TrackRef&) ;

    /// reconstruct trajectories from trajectory seed -- method for stand alone trajectory building
    MuonTrajectoryBuilder::TrajectoryContainer trajectories(const TrajectorySeed&) { MuonTrajectoryBuilder::TrajectoryContainer result; return result; }

    /// pass the Event Setup to the algo at each event
    virtual void setES(const edm::EventSetup&);
  
    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

  private:

    std::vector<reco::TrackRef> chooseRegionalTrackerTracks(const reco::TrackRef&, const edm::Handle<reco::TrackCollection>& ) const;

    RectangularEtaPhiTrackingRegion defineRegionOfInterest(const reco::TrackRef&) const;

    /// build combined trajectory from sta Track and tracker RecHits
    MuonTrajectoryBuilder::CandidateContainer build(const reco::TrackRef&, const std::vector<reco::TrackRef>&) const;
  
    /// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
    void checkMuonHits(const reco::Track&, ConstRecHitContainer&, ConstRecHitContainer&, std::vector<int>&) const;
 
    /// select muon hits compatible with trajectory; check hits in chambers with showers
    ConstRecHitContainer selectMuonHits(const Trajectory&, const std::vector<int>&) const;
 
    /// get TransientTrackingRecHits from Muon Track
    ConstMuonRecHitContainer getTransientHits(const reco::Track&) const;

    /// get TransientTrackingRecHits from Tracker Track
    ConstRecHitContainer getTkTransientHits(const reco::Track&) const;

    /// choose final trajectory
    const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&) const;

    /// calculate chi2 probability (-ln(P))
    double trackProbability(const Trajectory&) const;    

    /// get silicon tracker Trajectories from track Track
    TC getTrajFromTrack(const reco::TrackRef&) const;

    //// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

  private:

    GlobalPoint theVertexPos;
    GlobalError theVertexErr;
    MuonUpdatorAtVertex* theUpdator;
    GlobalMuonTrackMatcher* theTrackMatcher;
    GlobalMuonReFitter* theRefitter;
    MuonDetLayerMeasurements* theLayerMeasurements;

    float theTrackMatcherChi2Cut;
    int   theMuonHitsOption;
    ReconstructionDirection theDirection;
    float thePtCut;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;

    std::string thePropagatorLabel;
    std::string theTTRHBuilderName;   

    std::string theTkTrackLabel;

    edm::ESHandle<MagneticField> theField;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;

    edm::Handle<reco::TrackCollection> allTrackerTracks;

    edm::ESHandle<TransientTrackingRecHitBuilder> theTkTransientTrackingRecHitBuilder;

};
#endif
