
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <limits>
#include <algorithm>
#include "helper.h"

template< typename PATOBJ >
class MatchEmbedder : public edm::global::EDProducer<> {

  // perhaps we need better structure here (begin run etc)


public:

  explicit MatchEmbedder(const edm::ParameterSet &cfg):
    src_{consumes<PATOBJCollection>(cfg.getParameter<edm::InputTag>("src"))},
    matching_{consumes< edm::Association<reco::GenParticleCollection> >( cfg.getParameter<edm::InputTag>("matching") )} {
    produces<PATOBJCollection>();
  }

  ~MatchEmbedder() override {}

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {}

private:
  typedef std::vector<PATOBJ> PATOBJCollection;
  const edm::EDGetTokenT<PATOBJCollection> src_;
  const edm::EDGetTokenT< edm::Association<reco::GenParticleCollection> > matching_;
};

template< typename PATOBJ >
void MatchEmbedder<PATOBJ>::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const & iSetup) const {

  //input
  edm::Handle<PATOBJCollection> src;
  evt.getByToken(src_, src);

  edm::Handle< edm::Association<reco::GenParticleCollection> > matching;
  evt.getByToken(matching_, matching);

  size_t nsrc = src->size();
  // output
  std::unique_ptr<PATOBJCollection>  out(new PATOBJCollection() );
  out->reserve(nsrc);

  for (unsigned int i = 0; i < nsrc; ++i) {
    edm::Ptr<PATOBJ> ptr(src, i);
    reco::GenParticleRef match = (*matching)[ptr];
    out->emplace_back(src->at(i));
    out->back().addUserInt(
      "mcMatch",
      match.isNonnull() ? match->pdgId() : 0
    );
  }

  //adding label to be consistent with the muon and track naming
  evt.put(std::move(out));
}

#include "DataFormats/PatCandidates/interface/Muon.h"
typedef MatchEmbedder<pat::Muon> MuonMatchEmbedder;

#include "DataFormats/PatCandidates/interface/Electron.h"
typedef MatchEmbedder<pat::Electron> ElectronMatchEmbedder;

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
typedef MatchEmbedder<pat::CompositeCandidate> CompositeCandidateMatchEmbedder;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonMatchEmbedder);
DEFINE_FWK_MODULE(ElectronMatchEmbedder);
DEFINE_FWK_MODULE(CompositeCandidateMatchEmbedder);
