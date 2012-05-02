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

  bool vxTrkFiltering;

// Get the highest pt track in a jet.
// Get the KF track if it exists.  Otherwise, see if it has a GSF track.
  reco::TrackBaseRef RecoTauVertexAssociator::getLeadTrack(const PFJet& jet) const{
  std::vector<PFCandidatePtr> allTracks = pfChargedCands(jet, true);
  std::vector<PFCandidatePtr> tracks;
  //PJ filtering of tracks 
 if(vxTrkFiltering) tracks = qcuts_.filterRefs(allTracks);
  else tracks = allTracks;
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
  namespace {
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
						 const edm::ParameterSet& pset):  qcuts_(pset.getParameterSet("vxAssocQualityCuts"))
 {
  vertexTag_ = edm::InputTag("offlinePrimaryVertices", "");
  std::string algorithm = "highestPtInEvent";
  // Sanity check, will remove once HLT module configs are updated.
  if (!pset.exists("primaryVertexSrc") || !pset.exists("pvFindingAlgo")) {
    edm::LogWarning("NoVertexFindingMethodSpecified")
      << "The PSet passed to the RecoTauVertexAssociator was"
      << " incorrectly configured. The vertex will be taken as the "
      << "highest Pt vertex from the offlinePrimaryVertices collection."
      << std::endl;
  } else {
    vertexTag_ = pset.getParameter<edm::InputTag>("primaryVertexSrc");
    algorithm = pset.getParameter<std::string>("pvFindingAlgo");
  }
  vxTrkFiltering = false;
  if(!pset.exists("vertexTrackFiltering")){
       edm::LogWarning("NoVertexTrackFilteringSpecified")
	 << "The PSet passed to the RecoTauVertexAssociator was"
	 << " incorrectly configured. Please define vertexTrackFiltering in config file." 
	 << " No filtering of tracks to vertices will be applied"
	 << std::endl;
     }else{
       vxTrkFiltering = pset.getParameter<bool>("vertexTrackFiltering");
     }
 
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
  if(vertices_.size()>0 ) qcuts_.setPV(vertices_[0]);
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFTau& tau) const {
  reco::PFJetRef jetRef = tau.jetRef();
  // FIXME workaround for HLT which does not use updated data format
  if (jetRef.isNull())
    jetRef = tau.pfTauTagInfoRef()->pfjetRef();

  return associatedVertex(*jetRef);
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFJet& jet) const {
  reco::VertexRef output = vertices_.size() ? vertices_[0] : reco::VertexRef();
  if (algo_ == kHighestPtInEvent) {
    return output;
  } else if (algo_ == kClosestDeltaZ) {
    double closestDistance = std::numeric_limits<double>::infinity();
    DZtoTrack dzComputer(getLeadTrack(jet));
    // Find the vertex that has the lowest DZ to the lead track
    BOOST_FOREACH(const reco::VertexRef& vtx, vertices_) {
      double dz = dzComputer(vtx);
      if (dz < closestDistance) {
        closestDistance = dz;
        output = vtx;
      }
    }
  } else if (algo_ == kHighestWeigtForLeadTrack) {
    double largestWeight = 0.;
    // Find the vertex that gives the lead track the highest weight. 
    TrackWeightInVertex weightComputer(getLeadTrack(jet));
    // Find the vertex that has the lowest DZ to the lead track
    BOOST_FOREACH(const reco::VertexRef& vtx, vertices_) {
      double weight = weightComputer(vtx);
      if (weight > largestWeight) {
        largestWeight = weight;
        output = vtx;
      }
    }
  }
  return output;
}

}}
