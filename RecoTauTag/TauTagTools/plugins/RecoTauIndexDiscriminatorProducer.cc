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

namespace {

class RecoTauIndexDiscriminatorProducer final : public PFTauDiscriminationProducerBase {
  public:
      explicit RecoTauIndexDiscriminatorProducer(const edm::ParameterSet& cfg)
        :PFTauDiscriminationProducerBase(cfg) {}
      ~RecoTauIndexDiscriminatorProducer() override{}
      double discriminate(const reco::PFTauRef& thePFTauRef) const override;
      void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup) override {};
};

double RecoTauIndexDiscriminatorProducer::discriminate(const reco::PFTauRef& thePFTauRef) const {
  return thePFTauRef.key();
}

}

DEFINE_FWK_MODULE(RecoTauIndexDiscriminatorProducer);
