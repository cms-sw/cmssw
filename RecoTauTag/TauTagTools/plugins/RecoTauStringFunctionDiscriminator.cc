/*
 * class RecoTauStringFunctionDiscriminator
 * Author : Evan K. Friis (UC Davis)
 *
 * Helper utility module that produces a PFTauDiscriminator
 * that stores the result of a string function on a tau.
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

class RecoTauStringFunctionDiscriminator :
  public PFTauDiscriminationProducerBase {
  public:

      explicit RecoTauStringFunctionDiscriminator(const edm::ParameterSet& cfg)
        :PFTauDiscriminationProducerBase(cfg),
        function_(cfg.getParameter<std::string>("function")) {}

      ~RecoTauStringFunctionDiscriminator(){}

      double discriminate(const reco::PFTauRef& thePFTauRef);

  private:
      StringObjectFunction<reco::PFTau> function_;
};

double RecoTauStringFunctionDiscriminator::discriminate(const reco::PFTauRef& thePFTauRef) {
  return function_(*thePFTauRef);
}

DEFINE_FWK_MODULE(RecoTauStringFunctionDiscriminator);
