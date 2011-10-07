/*
 * class RecoTauHashDiscriminatorProducer
 * Author : Evan K. Friis (UC Davis)
 *
 * Helper utility module that produces a PFTauDiscriminator
 * that only contains a unique identifier for a PFTau.
 *
 * Currently, it is only the index into the original colleciton.
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

class RecoTauIndexDiscriminatorProducer : public PFTauDiscriminationProducerBase {
  public:
      explicit RecoTauIndexDiscriminatorProducer(const edm::ParameterSet& cfg)
        :PFTauDiscriminationProducerBase(cfg) {}
      ~RecoTauIndexDiscriminatorProducer(){}
      double discriminate(const reco::PFTauRef& thePFTauRef);
      void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup) {};
};

double RecoTauIndexDiscriminatorProducer::discriminate(const reco::PFTauRef& thePFTauRef) {
  return thePFTauRef.key();
}

DEFINE_FWK_MODULE(RecoTauIndexDiscriminatorProducer);
