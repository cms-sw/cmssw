#ifndef RecoMuon_L3TrackFinder_L3MuonTrajectoryBuilder_H
#define RecoMuon_L3TrackFinder_L3MuonTrajectoryBuilder_H

/** \class L3MuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  \author N. Neumeister   Purdue University
 *  \author C. Liu          Purdue University
 *  \author A. Everett      Purdue University
 */

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonServiceProxy;
class Trajectory;
class TrajectoryCleaner;

class L3MuonTrajectoryBuilder : public GlobalTrajectoryBuilderBase {

  public:

    /// Constructor with Parameter Set and MuonServiceProxy
	L3MuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*, edm::ConsumesCollector&);

    /// Destructor
    ~L3MuonTrajectoryBuilder();

    /// Reconstruct trajectories from standalone and tracker only Tracks
    using GlobalTrajectoryBuilderBase::trajectories;
    MuonTrajectoryBuilder::CandidateContainer trajectories(const TrackCand&);

    /// Pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

    /// Add default values for fillDescriptions
    static void fillDescriptions(edm::ParameterSetDescription& descriptions);

  private:

    /// Make a TrackCand collection using tracker Track, Trajectory information
    std::vector<TrackCand> makeTkCandCollection(const TrackCand&);

    TrajectoryCleaner* theTrajectoryCleaner;
    edm::InputTag theTkCollName;
    edm::Handle<reco::TrackCollection> allTrackerTracks;
    reco::BeamSpot beamSpot;
    edm::Handle<reco::BeamSpot> beamSpotHandle;
    edm::InputTag theBeamSpotInputTag;
    reco::Vertex vtx;
    edm::Handle<reco::VertexCollection> pvHandle;
    edm::InputTag theVertexCollInputTag;
    bool theUseVertex;
    double theMaxChi2;
    double theDXYBeamSpot;
    edm::EDGetTokenT<reco::TrackCollection> theTrackToken;
};
#endif
