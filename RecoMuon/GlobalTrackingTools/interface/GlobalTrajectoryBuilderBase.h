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
 *
 *  $Date: 2008/02/14 20:35:07 $
 *  $Revision: 1.4 $
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

class RectangularEtaPhiTrackingRegion;
class TrajectoryStateOnSurface;
class GlobalMuonTrackMatcher;
class MuonDetLayerMeasurements;
class MuonServiceProxy;
class Trajectory;
class TrackTransformer;
class TrajectoryFitter;
class MuonTrackingRegionBuilder;

namespace edm {class ParameterSet; class Event;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

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

  public:

    /// constructor with Parameter Set and MuonServiceProxy
    GlobalTrajectoryBuilderBase(const edm::ParameterSet&, const MuonServiceProxy*);
          
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
 
    /// choose final trajectory
    const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&, int) const;

    /// calculate chi2 probability (-ln(P))
    double trackProbability(const Trajectory&) const;

    /// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

    /// if TrackCand has only a TrackRef, attempt to add Trajectory*
    void addTraj(TrackCand&);

    /// check order of RechIts on a trajectory
    RefitDirection checkRecHitsOrdering(const ConstRecHitContainer&) const;

    /// refit a trajectory
    std::vector<Trajectory> refitTrajectory(const Trajectory&) const;

    /// build a global trajectory from tracker and muon hits
    std::vector<Trajectory> glbTrajectory(const TrajectorySeed& seed,
                                          const ConstRecHitContainer& trackerhits,
                                          const ConstRecHitContainer& muonhits,
                                          const TrajectoryStateOnSurface& firstPredTsos) const ;

    ///
    GlobalMuonTrackMatcher* trackMatcher() const { return theTrackMatcher; }

    ///
    const MuonServiceProxy* service() const { return theService; }

  protected:

    std::string theCategory;
    bool theTkTrajsAvailableFlag;
    float thePtCut;

  private:

    GlobalMuonTrackMatcher* theTrackMatcher;
    MuonDetLayerMeasurements* theLayerMeasurements;
    TrackTransformer* theTrackTransformer;
    MuonTrackingRegionBuilder* theRegionBuilder;
    const MuonServiceProxy* theService;
    edm::ESHandle<TrajectoryFitter> theKFFitter;
  
    int   theMuonHitsOption;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;
    std::string theKFFitterName;
    std::string theTrackerPropagatorName;
 
    const edm::Event* theEvent;

};
#endif
