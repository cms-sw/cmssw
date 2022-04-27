#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/GenEvent.h"

class HepMCCopy : public edm::one::EDProducer<> {
public:
  explicit HepMCCopy(edm::ParameterSet const& p);
  ~HepMCCopy() override = default;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
};

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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HepMCCopy);
