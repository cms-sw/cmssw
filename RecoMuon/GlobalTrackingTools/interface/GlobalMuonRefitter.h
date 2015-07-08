#ifndef RecoMuon_GlobalTrackingTools_GlobalMuonRefitter_H
#define RecoMuon_GlobalTrackingTools_GlobalMuonRefitter_H

/** \class GlobalMuonRefitter
 *  class to build muon trajectory
 *
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 *  \modified by C. Calabria & A. Sharma    INFN & Universita  Bari
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/MuonReco/interface/DYTInfo.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

namespace edm {class Event;}
namespace reco {class TransientTrack;}

class TrajectoryStateOnSurface;
class TrackerTopology;

class MuonDetLayerMeasurements;
class MuonServiceProxy;
class Trajectory;

class TrajectoryFitter;

class GlobalMuonRefitter {

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

    enum subDetector { PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6 };

  public:

    /// constructor with Parameter Set and MuonServiceProxy
    GlobalMuonRefitter(const edm::ParameterSet&, const MuonServiceProxy*, edm::ConsumesCollector&);
          
    /// destructor
    virtual ~GlobalMuonRefitter();

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

    /// set the services needed by the TrackTransformer
    void setServices(const edm::EventSetup&);

    /// build combined trajectory from sta Track and tracker RecHits
    std::vector<Trajectory> refit(const reco::Track& globalTrack, const int theMuonHitsOption, 
				  const TrackerTopology *tTopo) const;

    /// build combined trajectory from subset of sta Track and tracker RecHits
    std::vector<Trajectory> refit(const reco::Track& globalTrack,
				  const reco::TransientTrack track,
				  const TransientTrackingRecHit::ConstRecHitContainer& allRecHitsTemp,
				  const int theMuonHitsOption,
				  const TrackerTopology *tTopo) const;

    /// refit the track with a new set of RecHits
    std::vector<Trajectory> transform(const reco::Track& newTrack,
                                      const reco::TransientTrack track,
                                      const TransientTrackingRecHit::ConstRecHitContainer& recHitsForReFit) const;
    
    // get rid of selected station RecHits
    ConstRecHitContainer getRidOfSelectStationHits(const ConstRecHitContainer& hits,
						   const TrackerTopology *tTopo) const;

    // return DYT-related informations           
    const reco::DYTInfo* getDYTInfo() {return dytInfo;}
    
  protected:

    enum RefitDirection{insideOut,outsideIn,undetermined};
    
    /// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
    void checkMuonHits(const reco::Track&, ConstRecHitContainer&, 
                       std::map<DetId, int> &) const;

    /// get the RecHits in the tracker and the first muon chamber with hits 
    void getFirstHits(const reco::Track&, ConstRecHitContainer&, 
                       ConstRecHitContainer&) const;
 
    /// select muon hits compatible with trajectory; check hits in chambers with showers
    ConstRecHitContainer selectMuonHits(const Trajectory&, 
                                        const std::map<DetId, int> &) const;
 
    /// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

    RefitDirection checkRecHitsOrdering(const ConstRecHitContainer&) const;

    const MuonServiceProxy* service() const { return theService; }

  protected:
    std::string theCategory;
    bool theTkTrajsAvailableFlag;
    float thePtCut;

  private:
  
    int   theMuonHitsOption;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;
    float theGEMChi2Cut;
    bool  theCosmicFlag;

    edm::InputTag theDTRecHitLabel;
    edm::InputTag theCSCRecHitLabel;
    edm::InputTag theGEMRecHitLabel;
    edm::Handle<DTRecHitCollection>    theDTRecHits;
    edm::Handle<CSCRecHit2DCollection> theCSCRecHits;
    edm::Handle<GEMRecHitCollection> theGEMRecHits;
    edm::EDGetTokenT<DTRecHitCollection> theDTRecHitToken;
    edm::EDGetTokenT<CSCRecHit2DCollection> theCSCRecHitToken;
    edm::EDGetTokenT<GEMRecHitCollection> theGEMRecHitToken;

    int	  theSkipStation;
    int   theTrackerSkipSystem;
    int   theTrackerSkipSection;

    unsigned long long theCacheId_TRH;        

    std::string thePropagatorName;
  
    bool theRPCInTheFit;

    double theRescaleErrorFactor;

    RefitDirection theRefitDirection;

    std::vector<int> theDYTthrs;
    int theDYTselector;
    bool theDYTupdator;
    bool theDYTuseAPE;
    reco::DYTInfo *dytInfo;

    std::string theFitterName;
    std::unique_ptr<TrajectoryFitter> theFitter;
  
    std::string theTrackerRecHitBuilderName;
    edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
    TkClonerImpl hitCloner;
  
    std::string theMuonRecHitBuilderName;
    edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;

    const MuonServiceProxy* theService;
    const edm::Event* theEvent;

    edm::EDGetTokenT<CSCSegmentCollection> CSCSegmentsToken;
    edm::EDGetTokenT<DTRecSegment4DCollection> all4DSegmentsToken;
    edm::Handle<CSCSegmentCollection> CSCSegments;
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
};
#endif
