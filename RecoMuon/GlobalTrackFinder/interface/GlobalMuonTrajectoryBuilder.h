#ifndef GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: 2006/08/28 19:32:58 $
 *  $Revision: 1.26 $
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
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
class TransientTrackingRecHitBuilder;
class MuonTransientTrackingRecHitBuilder;
class GlobalTrackingGeometry;
class MuonDetLayerGeometry;
class MuonDetLayerMeasurements;
class MuonTrackReFitter;
class TrackConverter;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder {

  public:

    typedef std::pair<Trajectory*, reco::TrackRef> TrackCand;
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

    /// constructor with Parameter Set
    GlobalMuonTrajectoryBuilder(const edm::ParameterSet&);
          
    /// destructor
    ~GlobalMuonTrajectoryBuilder();

    /// reconstruct trajectories from standalone and tracker only Tracks
    MuonTrajectoryBuilder::CandidateContainer trajectories(const reco::TrackRef&);

    /// reconstruct trajectories from trajectory seed
    MuonTrajectoryBuilder::TrajectoryContainer trajectories(const TrajectorySeed&){ MuonTrajectoryBuilder::TrajectoryContainer result; return result; }

    /// pass the Event Setup to the algo at each event
    virtual void setES(const edm::EventSetup&);
  
    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

  private:

    /// choose tracker tracks within region of interest
    std::vector<TrackCand> chooseRegionalTrackerTracks(const TrackCand&, 
                                                       const edm::Handle<reco::TrackCollection>&) const;

    /// define region of interest with tracker
    RectangularEtaPhiTrackingRegion defineRegionOfInterest(const reco::TrackRef&) const;

    /// build combined trajectory from sta Track and tracker RecHits
    MuonTrajectoryBuilder::CandidateContainer build(const TrackCand&, 
                                                    const std::vector<TrackCand>&) const;
  
    /// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
    void checkMuonHits(const reco::Track&, ConstRecHitContainer&, ConstRecHitContainer&, std::vector<int>&) const;
 
    /// select muon hits compatible with trajectory; check hits in chambers with showers
    ConstRecHitContainer selectMuonHits(const Trajectory&, const std::vector<int>&) const;
 
    /// choose final trajectory
    const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&) const;

    /// calculate chi2 probability (-ln(P))
    double trackProbability(const Trajectory&) const;    

    //// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

  private:

    GlobalPoint theVertexPos;
    GlobalError theVertexErr;
    MuonUpdatorAtVertex* theUpdator;
    GlobalMuonTrackMatcher* theTrackMatcher;
    MuonTrackReFitter* theRefitter;
    MuonDetLayerMeasurements* theLayerMeasurements;
    MuonTransientTrackingRecHitBuilder* theMuonTTRHBuilder;
    TrackConverter* theTrackConverter;

    float theTrackMatcherChi2Cut;
    int   theMuonHitsOption;
    ReconstructionDirection theDirection;
    float thePtCut;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;
    bool convert;

    std::string thePropagatorLabel;
    std::string theTTRHBuilderName;   

    std::string theTkTrackLabel;

    edm::ESHandle<MagneticField> theMagField;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;

    edm::Handle<reco::TrackCollection> allTrackerTracks;

    edm::ESHandle<TransientTrackingRecHitBuilder> theTkTransientTrackingRecHitBuilder;

};
#endif
