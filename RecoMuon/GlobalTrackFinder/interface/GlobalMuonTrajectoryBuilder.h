#ifndef RecoMuon_GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define RecoMuon_GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {class ParameterSet; class Event; class EventSetup; }

class MuonServiceProxy;
class Trajectory;

class GlobalMuonTrajectoryBuilder : public GlobalTrajectoryBuilderBase {

  public:

    /// constructor with Parameter Set and MuonServiceProxy
  GlobalMuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*,edm::ConsumesCollector&);
          
    /// destructor
    ~GlobalMuonTrajectoryBuilder() override;

    /// reconstruct trajectories from standalone and tracker only Tracks    
    MuonTrajectoryBuilder::CandidateContainer trajectories(const TrackCand&) override;

    /// pass the Event to the algo at each event
    void setEvent(const edm::Event&) override;

  private:
  
    /// make a TrackCand collection using tracker Track, Trajectory information
    std::vector<TrackCand> makeTkCandCollection(const TrackCand&) override;

  private:
  
    edm::InputTag theTkTrackLabel;
    edm::EDGetTokenT<reco::TrackCollection> allTrackerTracksToken;
    edm::Handle<reco::TrackCollection> allTrackerTracks;

};
#endif
