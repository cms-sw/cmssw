#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco { namespace tau {

namespace {
// Find the vertex in [vertices] that is closest in Z to the lead track of the
// jet.
reco::VertexRef closestVertex(
    const edm::Handle<reco::VertexCollection>& vertices,
    const reco::PFJet& jet) {
  // Take the first one if exists
  reco::VertexRef selectedVertex = vertices->size() ?
    reco::VertexRef(vertices, 0) : reco::VertexRef();
  double minDZ = std::numeric_limits<double>::infinity();
  // Find the lead charged object in the jet (true mean sort)
  std::vector<PFCandidatePtr> tracks = pfChargedCands(jet, true);
  if (!tracks.size())
    return selectedVertex;
  const reco::Track* leadTrack = NULL;

  if (tracks[0]->trackRef().isNonnull()) {
    leadTrack = tracks[0]->trackRef().get();
  } else if (tracks[0]->gsfTrackRef().isNonnull()) {
    const reco::GsfTrack* gsfTrack = tracks[0]->gsfTrackRef().get();
    leadTrack = static_cast<const reco::Track*>(gsfTrack);
  }
  // Try to get the gsfTrack if possible
  if (!leadTrack) {
    edm::LogError("NullTrackRefInLeadPFCharged")
      << "The leading *charged* PF cand in the jet has an invalid TrackRef!"
      << " LeadPFCand: " << *tracks[0];
    return selectedVertex;
  }
  // Loop over all the vertices and check if they are closer in z to the
  // current track.
  for (unsigned int ivtx = 0; ivtx < vertices->size(); ivtx++) {
    reco::VertexRef pvCand(vertices, ivtx);
    double dz = std::abs(leadTrack->dz(pvCand->position()));
    if (dz < minDZ) {
      minDZ = dz;
      selectedVertex = pvCand;
    }
  }
  return selectedVertex;
}
}

RecoTauVertexAssociator::RecoTauVertexAssociator(
    const edm::ParameterSet& pset) {
  if (!pset.exists("primaryVertexSrc") || !pset.exists("useClosestPV")) {
    edm::LogError("VertexAssociatorMisconfigured")
      << "The RecoTauVertexAssociator was not passed one of the"
      << " required arguments" << std::endl;
  }
  vertexTag_ = pset.getParameter<edm::InputTag>("primaryVertexSrc");
  useClosest_ = pset.getParameter<bool>("useClosestPV");
}

void RecoTauVertexAssociator::setEvent(const edm::Event& evt) {
  evt.getByLabel(vertexTag_, vertices_);
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFTau& tau) const {
  return associatedVertex(*tau.jetRef());
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFJet& jet) const {
  if (useClosest_) {
    return closestVertex(vertices_, jet);
  } else {
    // Just take the first vertex
    if (vertices_->size()) {
      return reco::VertexRef(vertices_, 0);
    }
  }
  return reco::VertexRef();
}

}}
