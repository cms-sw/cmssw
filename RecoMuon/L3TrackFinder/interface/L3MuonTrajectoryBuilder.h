#ifndef RecoMuon_L3TrackFinder_L3MuonTrajectoryBuilder_H
#define RecoMuon_L3TrackFinder_L3MuonTrajectoryBuilder_H

/** \class L3MuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonServiceProxy;
class Trajectory;
class TrajectoryCleaner;

class L3MuonTrajectoryBuilder : public GlobalTrajectoryBuilderBase {
  public:
    /// constructor with Parameter Set and MuonServiceProxy
  L3MuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*, edm::ConsumesCollector&);
          
    /// destructor
    ~L3MuonTrajectoryBuilder();
    /// reconstruct trajectories from standalone and tracker only Tracks    
    MuonTrajectoryBuilder::CandidateContainer trajectories(const TrackCand&);
    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);
  private:
    /// make a TrackCand collection using tracker Track, Trajectory information
    std::vector<TrackCand> makeTkCandCollection(const TrackCand&);
  private:
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
    edm::EDGetTokenT<reco::TrackCollection> trackToken_;

};
#endif
