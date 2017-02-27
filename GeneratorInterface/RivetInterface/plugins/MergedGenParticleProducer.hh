#ifndef MergedGenParticleProducer_h
#define MergedGenParticleProducer_h

#include <vector>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

class MergedGenParticleProducer : public edm::EDProducer {
 public:
  explicit MergedGenParticleProducer(const edm::ParameterSet &);
  ~MergedGenParticleProducer();

 private:
  virtual void beginJob();
  virtual void produce(edm::Event &, const edm::EventSetup &);
  virtual void endJob();

  edm::EDGetTokenT<edm::View<reco::GenParticle>> input_pruned_;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> input_packed_;
  reco::GenParticleRefProd ref_;
};

#endif
