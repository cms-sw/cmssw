#ifndef RecoMuon_GlobalTrackingTools_GlobalMuonRefitter_H
#define RecoMuon_GlobalTrackingTools_GlobalMuonRefitter_H

/** \class GlobalMuonRefitter
 *  class to build muon trajectory
 *
 *  $Date: 2008/04/29 13:49:47 $
 *  $Revision: 1.1 $
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

namespace edm {class Event;}
namespace reco {class TransientTrack;}

class TrajectoryStateOnSurface;

class MuonDetLayerMeasurements;
class MuonServiceProxy;
class Trajectory;

class TrackTransformer;
class TrajectoryFitter;

class GlobalMuonRefitter : public TrackTransformer {

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
    GlobalMuonRefitter(const edm::ParameterSet&, const MuonServiceProxy*);
          
    /// destructor
    virtual ~GlobalMuonRefitter();

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

    /// build combined trajectory from sta Track and tracker RecHits
    std::vector<Trajectory> refit(const reco::Track& globalTrack , const int theMuonHitsOption) const;

  protected:

    enum RefitDirection{inToOut,outToIn,undetermined};
    
    /// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
    void checkMuonHits(const reco::Track&, ConstRecHitContainer&, 
                       ConstRecHitContainer&, 
                       std::vector<int>&) const;
 
    /// select muon hits compatible with trajectory; check hits in chambers with showers
    ConstRecHitContainer selectMuonHits(const Trajectory&, 
                                        const std::vector<int>&) const;
 
    /// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

    RefitDirection checkRecHitsOrdering(const ConstRecHitContainer&) const;

    const MuonServiceProxy* service() const { return theService; }

  protected:
    std::string theCategory;
    bool theTkTrajsAvailableFlag;
    float thePtCut;

  private:

    MuonDetLayerMeasurements* theLayerMeasurements;
    const MuonServiceProxy* theService;
    TrackTransformer* theTrackTransformer;
  
    int   theMuonHitsOption;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;
 
    const edm::Event* theEvent;

};
#endif
