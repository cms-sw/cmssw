#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include <functional>
#include <boost/foreach.hpp>

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco { namespace tau {

namespace {

// Get the highest pt track in a jet.
// Get the KF track if it exists.  Otherwise, see if it has a GSF track.
const reco::TrackBaseRef getLeadTrack(const PFJet& jet) {
  std::vector<PFCandidatePtr> tracks = pfChargedCands(jet, true);
  if (!tracks.size())
    return reco::TrackBaseRef();
  PFCandidatePtr cand = tracks[0];
  if (cand->trackRef().isNonnull())
    return reco::TrackBaseRef(cand->trackRef());
  else if (cand->gsfTrackRef().isNonnull()) {
    return reco::TrackBaseRef(cand->gsfTrackRef());
  }
  return reco::TrackBaseRef();
}

// Define functors which extract the relevant information from a collection of
// vertices.
class DZtoTrack : public std::unary_function<double, reco::VertexRef> {
  public:
    DZtoTrack(const reco::TrackBaseRef& trk):trk_(trk){}
    double operator()(const reco::VertexRef& vtx) const {
      if (!trk_ || !vtx) {
        return std::numeric_limits<double>::infinity();
      }
      return std::abs(trk_->dz(vtx->position()));
    }
  private:
    const reco::TrackBaseRef trk_;
};

class TrackWeightInVertex : public std::unary_function<double, reco::VertexRef>
{
  public:
    TrackWeightInVertex(const reco::TrackBaseRef& trk):trk_(trk){}
    double operator()(const reco::VertexRef& vtx) const {
      if (!trk_ || !vtx) {
        return 0.0;
      }
      return vtx->trackWeight(trk_);
    }
  private:
    const reco::TrackBaseRef trk_;
};

}

RecoTauVertexAssociator::RecoTauVertexAssociator(
    const edm::ParameterSet& pset) {
  vertexTag_ = pset.getParameter<edm::InputTag>("primaryVertexSrc");
  std::string algorithm = pset.getParameter<std::string>("pvFindingAlgo");
  if (algorithm == "highestPtInEvent") {
    algo_ = kHighestPtInEvent;
  } else if (algorithm == "closestInDeltaZ") {
    algo_ = kClosestDeltaZ;
  } else if (algorithm == "highestWeightForLeadTrack") {
    algo_ = kHighestWeigtForLeadTrack;
  } else {
    throw cms::Exception("BadVertexAssociatorConfig")
      << "The algorithm specified for tau-vertex association "
      << algorithm << " is invalid. Options are: "  << std::endl
      <<  "highestPtInEvent,"
      <<  "closestInDeltaZ,"
      <<  "or highestWeightForLeadTrack." << std::endl;
  }
}

void RecoTauVertexAssociator::setEvent(const edm::Event& evt) {
  edm::Handle<reco::VertexCollection> verticesH_;
  evt.getByLabel(vertexTag_, verticesH_);
  vertices_.clear();
  vertices_.reserve(verticesH_->size());
  for(size_t i = 0; i < verticesH_->size(); ++i) {
    vertices_.push_back(reco::VertexRef(verticesH_, i));
  }
  if (!vertices_.size()) {
    edm::LogError("NoPV") << "There is no primary vertex in the event!!!"
      << std::endl;
  }
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFTau& tau) const {
  return associatedVertex(*tau.jetRef());
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFJet& jet) const {
  if (algo_ == kHighestPtInEvent) {
    if (vertices_.size())
      return vertices_[0];
    else
      return reco::VertexRef();
  } else if (algo_ == kClosestDeltaZ) {
    double closestDistance = std::numeric_limits<double>::infinity();
    reco::VertexRef closestVertex;
    DZtoTrack dzComputer(getLeadTrack(jet));
    // Find the vertex that has the lowest DZ to the lead track
    BOOST_FOREACH(const reco::VertexRef& vtx, vertices_) {
      double dz = dzComputer(vtx);
      if (dz < closestDistance) {
        closestDistance = dz;
        closestVertex = vtx;
      }
    }
    return closestVertex;
  } else if (algo_ == kHighestWeigtForLeadTrack) {
    double largestWeight = 0.;
    reco::VertexRef heaviestVertex;
    // Find the vertex that gives the lead track the highest weight.
    TrackWeightInVertex weightComputer(getLeadTrack(jet));
    // Find the vertex that has the lowest DZ to the lead track
    BOOST_FOREACH(const reco::VertexRef& vtx, vertices_) {
      double weight = weightComputer(vtx);
      if (weight > largestWeight) {
        largestWeight = weight;
        heaviestVertex = vtx;
      }
    }
    return heaviestVertex;
  }
  throw cms::Exception("BadVertexAssociatorConfig")
    << "No suitable vertex association algo was found." << std::endl;
}

}}
