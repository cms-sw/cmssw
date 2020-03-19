#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "RecoParticleFlow/Configuration/plugins/HepMCCopy.h"
#include "HepMC/GenEvent.h"

HepMCCopy::HepMCCopy(edm::ParameterSet const& p) {
  // This producer produces a HepMCProduct, a copy of the original one
  produces<edm::HepMCProduct>();
}

void HepMCCopy::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  edm::Handle<edm::HepMCProduct> theHepMCProduct;
  bool source = iEvent.getByLabel("generatorSmeared", theHepMCProduct);
  if (!source) {
    auto pu_product = std::make_unique<edm::HepMCProduct>();
    iEvent.put(std::move(pu_product));
  } else {
    auto pu_product = std::make_unique<edm::HepMCProduct>(*theHepMCProduct);
    iEvent.put(std::move(pu_product));
  }
}

DEFINE_FWK_MODULE(HepMCCopy);
