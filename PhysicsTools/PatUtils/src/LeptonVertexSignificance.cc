#include "PhysicsTools/PatUtils/interface/LeptonVertexSignificance.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

using namespace pat;

edm::InputTag LeptonVertexSignificance::vertexCollectionTag() {
  return edm::InputTag("offlinePrimaryVerticesFromCTFTracks");
}

// calculate the TrackIsoPt for the lepton object
float LeptonVertexSignificance::calculate(const Electron& theElectron,
                                          const reco::VertexCollection& vertices,
                                          const TransientTrackBuilder& builder) {
  return this->calculate(*theElectron.gsfTrack(), vertices, builder);
}

float LeptonVertexSignificance::calculate(const Muon& theMuon,
                                          const reco::VertexCollection& vertices,
                                          const TransientTrackBuilder& builder) {
  return this->calculate(*theMuon.track(), vertices, builder);
}

// calculate the TrackIsoPt for the lepton's track
float LeptonVertexSignificance::calculate(const reco::Track& theTrack,
                                          const reco::VertexCollection& vertices,
                                          const TransientTrackBuilder& builder) {
  if (vertices.empty())
    return 0;
  reco::Vertex const& theVertex = vertices.front();
  // calculate the track-vertex association significance
  reco::TransientTrack theTrTrack = builder.build(&theTrack);
  GlobalPoint theVertexPoint(theVertex.position().x(), theVertex.position().y(), theVertex.position().z());
  FreeTrajectoryState theLeptonNearVertex = theTrTrack.trajectoryStateClosestToPoint(theVertexPoint).theState();
  return fabs(theVertex.position().z() - theLeptonNearVertex.position().z()) /
         sqrt(std::pow(theVertex.zError(), 2) + theLeptonNearVertex.cartesianError().position().czz());
}
