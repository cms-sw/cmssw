#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

namespace SmearedJetProducer_namespace
{
  template <>
  class GenJetMatcherT<pat::Jet>
  {
    public:

     GenJetMatcherT(const edm::ParameterSet&) {}
     ~GenJetMatcherT() {}

     const reco::GenJet* operator()(const pat::Jet& jet, edm::Event* evt = 0) const
     {
       return jet.genJet();
     }
  };
}

typedef SmearedJetProducerT<pat::Jet> SmearedPATJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SmearedPATJetProducer);
