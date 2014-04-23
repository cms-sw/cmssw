/*
 * Select reco:Vertices consistent with pileup.
 *
 * Author: Evan K. Friis
 *
 */


#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <functional>
#include <algorithm>

namespace {

class VertexTrackPtSumFilter : public std::unary_function<reco::Vertex, bool> {
  public:
    VertexTrackPtSumFilter(double minPt):minPt_(minPt){}
    bool operator()(const reco::Vertex& vtx) const {
      double trackPtSum = 0.;
      for ( reco::Vertex::trackRef_iterator track = vtx.tracks_begin();
          track != vtx.tracks_end(); ++track ) {
        trackPtSum += (*track)->pt();
      }
      return trackPtSum > minPt_;
    }
  private:
    double minPt_;
};

}

class RecoTauPileUpVertexSelector : public edm::stream::EDFilter<> {
  public:
    explicit RecoTauPileUpVertexSelector(const edm::ParameterSet &pset);
    ~RecoTauPileUpVertexSelector() {}
    bool filter(edm::Event& evt, const edm::EventSetup& es) override;
  private:
    edm::InputTag src_;
    VertexTrackPtSumFilter vtxFilter_;
    bool filter_;
    edm::EDGetTokenT<reco::VertexCollection> token;
};

RecoTauPileUpVertexSelector::RecoTauPileUpVertexSelector(
    const edm::ParameterSet& pset):vtxFilter_(
      pset.getParameter<double>("minTrackSumPt")) {
  src_ = pset.getParameter<edm::InputTag>("src");
  token = consumes<reco::VertexCollection>(src_);
  filter_ = pset.exists("filter") ? pset.getParameter<bool>("filter") : false;
  produces<reco::VertexCollection>();
}


bool RecoTauPileUpVertexSelector::filter(
    edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::VertexCollection> vertices_;
  evt.getByToken(token, vertices_);
  std::auto_ptr<reco::VertexCollection> output(new reco::VertexCollection);
  // If there is only one vertex, there are no PU vertices!
  if (vertices_->size() > 1) {
    // Copy over all the vertices that have associatd tracks with pt greater
    // than the threshold.  The predicate function is the VertexTrackPtSumFilter
    // better name: copy_if_not
    std::remove_copy_if(vertices_->begin()+1, vertices_->end(),
        std::back_inserter(*output), std::not1(vtxFilter_));
  }
  size_t nPUVtx = output->size();
  evt.put(output);
  // If 'filter' is enabled, return whether true if there are PU vertices
  if (!filter_)
    return true;
  else
    return nPUVtx;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPileUpVertexSelector);
