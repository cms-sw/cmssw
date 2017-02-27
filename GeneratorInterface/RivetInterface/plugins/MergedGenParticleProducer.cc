#include "../plugins/MergedGenParticleProducer.hh"
#include <vector>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


MergedGenParticleProducer::MergedGenParticleProducer(const edm::ParameterSet& config){
  input_pruned_ = consumes<edm::View<reco::GenParticle>>(config.getParameter<edm::InputTag>("inputPruned"));
  input_packed_ = consumes<edm::View<pat::PackedGenParticle>>(config.getParameter<edm::InputTag>("inputPacked"));
  produces<reco::GenParticleCollection>();
}

MergedGenParticleProducer::~MergedGenParticleProducer() { }

void MergedGenParticleProducer::produce(edm::Event& event,
                                    const edm::EventSetup& setup) {

  // Need a ref to the product now for creating the mother/daughter refs
  ref_ = event.getRefBeforePut<reco::GenParticleCollection>();

  // Get the input collections
  edm::Handle<edm::View<reco::GenParticle> > pruned_handle;
  event.getByToken(input_pruned_, pruned_handle);

  edm::Handle<edm::View<pat::PackedGenParticle> > packed_handle;
  event.getByToken(input_packed_, packed_handle);

  // First determine which packed particles are also still in the pruned collection
  // so that we can skip them later
  std::map<pat::PackedGenParticle const*, reco::GenParticle const*> st1_dup_map;

  // Also map pointers in the original pruned collection to their index in the vector.
  // This index will be the same in the merged collection.
  std::map<reco::Candidate const*, std::size_t> pruned_idx_map;

  for (unsigned i = 0; i < pruned_handle->size(); ++i) {
    reco::GenParticle const& src = pruned_handle->at(i);
    pruned_idx_map[&src] = i;
    if (src.status() == 1) {
      // Convert the pruned GenParticle into a PackedGenParticle then do an exact
      // floating point comparison of the pt, eta and phi
      // This of course relies on the PackedGenParticle constructor being identical
      // between the CMSSW version this sample was produced with and the one we're
      // analysing with
      pat::PackedGenParticle pks(src, reco::GenParticleRef());
      unsigned found_matches = 0;
      for (unsigned j = 0; j < packed_handle->size(); ++j) {
        pat::PackedGenParticle const& pk = packed_handle->at(j);
        if (pks.pdgId() == pk.pdgId()
           && pks.pt()  == pk.pt()
           && pks.eta() == pk.eta()
           && pks.phi() == pk.phi()) {
          ++found_matches;
        st1_dup_map[&pk] = &src;
        }
      }
      if (found_matches > 1) {
        std::cerr << "Warning, found multiple packed matches for: " << i << "\t" << src.pdgId() << "\t" << src.pt() << "\t" << src.y() << "\n";
      }
      if (found_matches == 0 && std::abs(src.y()) < 6.0) {
        std::cerr << "Warning, unmatched status 1: " << i << "\t" << src.pdgId() << "\t" << src.pt() << "\t" << src.y() << "\n";
      }
    }
  }

  // At this point we know what the size of the merged GenParticle will be so we can create it
  unsigned n = pruned_handle->size() + (packed_handle->size() - st1_dup_map.size());
  auto cands = std::unique_ptr<reco::GenParticleCollection>(new reco::GenParticleCollection(n));

  // First copy in all the pruned candidates
  for (unsigned i = 0; i < pruned_handle->size(); ++i) {
    reco::GenParticle const& old_cand = pruned_handle->at(i);
    reco::GenParticle & new_cand = cands->at(i);
    new_cand = reco::GenParticle(pruned_handle->at(i));
    // Update the mother and daughter refs to this new merged collection
    new_cand.resetMothers(ref_.id());
    new_cand.resetDaughters(ref_.id());
    for (unsigned m = 0; m < old_cand.numberOfMothers(); ++m) {
      new_cand.addMother(reco::GenParticleRef(ref_, pruned_idx_map.at(old_cand.mother(m))));
    }
    for (unsigned d = 0; d < old_cand.numberOfDaughters(); ++d) {
      new_cand.addDaughter(reco::GenParticleRef(ref_, pruned_idx_map.at(old_cand.daughter(d))));
    }
  }

  // Now copy in the packed candidates that are not already in the pruned
  for (unsigned i = 0, idx = pruned_handle->size(); i < packed_handle->size(); ++i) {
    pat::PackedGenParticle const& pk = packed_handle->at(i);
    if (st1_dup_map.count(&pk)) continue;
    reco::GenParticle & new_cand = cands->at(idx);
    new_cand = reco::GenParticle(pk.charge(), pk.p4(), pk.vertex(), pk.pdgId(), 1, true);
    for (unsigned m = 0; m < pk.numberOfMothers(); ++m) {
      new_cand.addMother(reco::GenParticleRef(ref_, pruned_idx_map.at(pk.mother(m))));
      // Since the packed candidates drop the vertex position we'll take this from the mother
      if (m == 0) {
        new_cand.setVertex(pk.mother(m)->vertex());
      }
      // Should then add this GenParticle as a daughter of its mother
      cands->at(pruned_idx_map.at(pk.mother(m))).addDaughter(reco::GenParticleRef(ref_, idx));
    }
    ++idx;
  }

  event.put(std::move(cands));
}

void MergedGenParticleProducer::beginJob() {

}

void MergedGenParticleProducer::endJob() {}

DEFINE_FWK_MODULE(MergedGenParticleProducer);
