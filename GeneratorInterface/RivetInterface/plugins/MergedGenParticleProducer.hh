#ifndef MergedGenParticleProducer_h
#define MergedGenParticleProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include <vector>

class MergedGenParticleProducer : public edm::stream::EDProducer<> {
public:
  MergedGenParticleProducer(const edm::ParameterSet& pset);
  ~MergedGenParticleProducer() override{};

private:
  void produce(edm::Event& event, const edm::EventSetup&) override;
  bool isPhotonFromPrunedHadron(const pat::PackedGenParticle& pk) const;
  bool isLeptonFromPrunedPhoton(const reco::GenParticle& pk) const;

  edm::EDGetTokenT<edm::View<reco::GenParticle>> input_pruned_;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> input_packed_;
};

#endif
