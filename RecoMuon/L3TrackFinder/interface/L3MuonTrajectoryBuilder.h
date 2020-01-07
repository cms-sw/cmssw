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
#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class MuonServiceProxy;
class Trajectory;
class TrajectoryCleaner;

class L3MuonTrajectoryBuilder : public GlobalTrajectoryBuilderBase {
public:
  /// Constructor with Parameter Set and MuonServiceProxy
  L3MuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*, edm::ConsumesCollector&);

  /// Destructor
  ~L3MuonTrajectoryBuilder() override;

  /// Reconstruct trajectories from standalone and tracker only Tracks
  using GlobalTrajectoryBuilderBase::trajectories;
  MuonTrajectoryBuilder::CandidateContainer trajectories(const TrackCand&) override;

  /// Pass the Event to the algo at each event
  void setEvent(const edm::Event&) override;

  /// Add default values for fillDescriptions
  static void fillDescriptions(edm::ParameterSetDescription& descriptions);

private:
  /// Make a TrackCand collection using tracker Track, Trajectory information
  std::vector<TrackCand> makeTkCandCollection(const TrackCand&) override;

  std::unique_ptr<TrajectoryCleaner> theTrajectoryCleaner;
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
