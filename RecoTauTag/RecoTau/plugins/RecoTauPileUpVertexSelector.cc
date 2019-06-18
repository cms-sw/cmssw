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

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <algorithm>

class RecoTauPileUpVertexSelector : public edm::stream::EDFilter<> {
public:
  explicit RecoTauPileUpVertexSelector(const edm::ParameterSet& pset);
  ~RecoTauPileUpVertexSelector() override {}
  bool filter(edm::Event& evt, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag src_;
  double minPt_;
  bool filter_;
  edm::EDGetTokenT<reco::VertexCollection> token;
};

RecoTauPileUpVertexSelector::RecoTauPileUpVertexSelector(const edm::ParameterSet& pset)
    : minPt_(pset.getParameter<double>("minTrackSumPt")) {
  src_ = pset.getParameter<edm::InputTag>("src");
  token = consumes<reco::VertexCollection>(src_);
  filter_ = pset.getParameter<bool>("filter");
  produces<reco::VertexCollection>();
}

bool RecoTauPileUpVertexSelector::filter(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::VertexCollection> vertices_;
  evt.getByToken(token, vertices_);
  auto output = std::make_unique<reco::VertexCollection>();
  // If there is only one vertex, there are no PU vertices!
  if (vertices_->size() > 1) {
    // Copy over all the vertices that have associatd tracks with pt greater
    // than the threshold.  The predicate function is the VertexTrackPtSumFilter
    // better name: copy_if_not
    std::remove_copy_if(vertices_->begin() + 1, vertices_->end(), std::back_inserter(*output), [this](auto const& vtx) {
      double trackPtSum = 0.;
      for (reco::Vertex::trackRef_iterator track = vtx.tracks_begin(); track != vtx.tracks_end(); ++track) {
        trackPtSum += (*track)->pt();
      }
      return trackPtSum > this->minPt_;
    });
  }
  size_t nPUVtx = output->size();
  evt.put(std::move(output));
  // If 'filter' is enabled, return whether true if there are PU vertices
  if (!filter_)
    return true;
  else
    return nPUVtx;
}

void RecoTauPileUpVertexSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // recoTauPileUpVertexSelector
  edm::ParameterSetDescription desc;

  desc.add<double>("minTrackSumPt");
  desc.add<edm::InputTag>("src");
  desc.add<bool>("filter");

  descriptions.add("recoTauPileUpVertexSelector", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPileUpVertexSelector);
