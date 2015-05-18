#ifndef RecoMuon_GlobalTrackingTools_GlobalTrajectoryBuilderBase_H
#define RecoMuon_GlobalTrackingTools_GlobalTrajectoryBuilderBase_H

/** \class GlobalTrajectoryBuilderBase
 *
 *   Base class for GlobalMuonTrajectoryBuilder and L3MuonTrajectoryBuilder
 *   Provide common tools and interface to reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information).
 *   It tries to reconstruct the corresponding
 *   track in the tracker and performs
 *   matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

//Forward declare classes from same package (RecoMuon)
class GlobalMuonTrackMatcher;
class MuonDetLayerMeasurements;
class MuonServiceProxy;
class MuonTrackingRegionBuilder;
class GlobalMuonRefitter;

class GlobalTrajectoryBuilderBase : public MuonTrajectoryBuilder {

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

    /// constructor with Parameter Set and MuonServiceProxy
    GlobalTrajectoryBuilderBase(const edm::ParameterSet&, const MuonServiceProxy*, edm::ConsumesCollector&);
          
    /// destructor
    virtual ~GlobalTrajectoryBuilderBase();

    /// dummy implementation, unused in this class
    MuonTrajectoryBuilder::TrajectoryContainer trajectories(const TrajectorySeed&) { return MuonTrajectoryBuilder::TrajectoryContainer(); }

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

  protected:

    enum RefitDirection{inToOut,outToIn,undetermined};

    /// build combined trajectory from sta Track and tracker RecHits
    MuonTrajectoryBuilder::CandidateContainer build(const TrackCand&,
                                                    MuonTrajectoryBuilder::CandidateContainer&) const;

    /// make a TrackCand collection using tracker Track, Trajectory information
    virtual std::vector<TrackCand> makeTkCandCollection(const TrackCand&) = 0;

    /// choose tracker tracks within region of interest
    std::vector<TrackCand> chooseRegionalTrackerTracks(const TrackCand&, 
                                                       const std::vector<TrackCand>&);

    /// define region of interest with tracker
    RectangularEtaPhiTrackingRegion defineRegionOfInterest(const reco::TrackRef&) const;

    /// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
    void checkMuonHits(const reco::Track&, 
                       ConstRecHitContainer&, 
                       ConstRecHitContainer&, 
                       std::vector<int>&) const;
 
    /// select muon hits compatible with trajectory; check hits in chambers with showers
    ConstRecHitContainer selectMuonHits(const Trajectory&, 
                                        const std::vector<int>&) const;

    /// select tracker hits; exclude some tracker hits in the global trajectory 
    ConstRecHitContainer selectTrackerHits(const ConstRecHitContainer&) const;

    /// rescale errors of outermost TEC RecHit
    void fixTEC(ConstRecHitContainer& all,
                double scl_x,
                double scl_y) const;

    /// choose final trajectory
    const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&, int) const;

    /// calculate chi2 probability (-ln(P))
    double trackProbability(const Trajectory&) const;

    /// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

    /// if TrackCand has only a TrackRef, attempt to add Trajectory*
    void addTraj(TrackCand&){} ///This does nothing now

    /// check order of RechIts on a trajectory
    RefitDirection checkRecHitsOrdering(const ConstRecHitContainer&) const;

    /// get transient RecHits of a Track
    TransientTrackingRecHit::ConstRecHitContainer
    getTransientRecHits(const reco::Track&) const;

    ///
    GlobalMuonTrackMatcher* trackMatcher() const { return theTrackMatcher; }

    ///
    const MuonServiceProxy* service() const { return theService; }

    struct ComparatorInOut {

      bool operator()(const TransientTrackingRecHit::ConstRecHitPointer& a,
		      const TransientTrackingRecHit::ConstRecHitPointer& b) const{ 
	bool barrel_a = ( a->det()->subDetector() == GeomDetEnumerators::DT ||
			  a->det()->subDetector() == GeomDetEnumerators::RPCBarrel );
	
	bool barrel_b = ( b->det()->subDetector() == GeomDetEnumerators::DT ||
			  b->det()->subDetector() == GeomDetEnumerators::RPCBarrel );
	
	 if ( barrel_a && barrel_b ) return  a->det()->surface().position().perp() < b->det()->surface().position().perp();

	else if ( !barrel_a && !barrel_b ) return  fabs(a->globalPosition().z()) < fabs(b->globalPosition().z());
	else if ( barrel_a && !barrel_b  ) return true;
	else if ( !barrel_a && barrel_b  ) return false;
	 //shouldn't really get here in any case (there's some sense to throw here )
	 return false;
      }
    };

    std::string theCategory;
    float thePtCut;
    float thePCut;

  private:

    GlobalMuonTrackMatcher* theTrackMatcher;
    MuonDetLayerMeasurements* theLayerMeasurements;
    TrackTransformer* theTrackTransformer;
    MuonTrackingRegionBuilder* theRegionBuilder;
    const MuonServiceProxy* theService;
    GlobalMuonRefitter* theGlbRefitter;
    unsigned long long theCacheId_TRH;
    bool theRPCInTheFit;
    bool  theRefitFlag;
    int   theMuonHitsOption;
    float theTECxScale;
    float theTECyScale;
    std::string theTrackerPropagatorName;
    const edm::Event* theEvent;
    std::string theTrackerRecHitBuilderName;
    edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
    std::string theMuonRecHitBuilderName;
    edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
    const TrackerTopology *theTopo;
};
#endif
