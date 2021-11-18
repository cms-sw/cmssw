#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace SmearedJetProducer_namespace {
  // template function to apply JER
  template <typename T>
  void SmearJet(T& jet, float smearfactor) {
    jet.scaleEnergy(smearfactor);
  }
  // template specialization for pat::Jets to store the JER factor
  template <>
  void SmearJet<pat::Jet>(pat::Jet& jet, float smearfactor) {
    jet.scaleEnergy(smearfactor);
    jet.addUserFloat("SmearFactor", smearfactor);
  }
}  // namespace SmearedJetProducer_namespace

#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

typedef SmearedJetProducerT<reco::CaloJet> SmearedCaloJetProducer;
typedef SmearedJetProducerT<reco::PFJet> SmearedPFJetProducer;
typedef SmearedJetProducerT<pat::Jet> SmearedPATJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SmearedCaloJetProducer);
DEFINE_FWK_MODULE(SmearedPFJetProducer);
DEFINE_FWK_MODULE(SmearedPATJetProducer);
