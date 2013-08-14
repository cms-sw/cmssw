/*
 * Produce a tau discriminator that produces a random discriminant output,
 * useful for testing.
 *
 * Author: Evan Friis, UC Davis
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "TRandom3.h"

class PFTauRandomDiscriminator : public PFTauDiscriminationProducerBase {
  public:
    PFTauRandomDiscriminator(const edm::ParameterSet& pset):
      PFTauDiscriminationProducerBase(pset) {
        passRate_ = pset.getParameter<double>("passRate");
      }

    double discriminate(const reco::PFTauRef& tau) override {
      return randy_.Rndm() < passRate_;
    }
  private:
    TRandom3 randy_;
    double passRate_;
};

DEFINE_FWK_MODULE(PFTauRandomDiscriminator);
