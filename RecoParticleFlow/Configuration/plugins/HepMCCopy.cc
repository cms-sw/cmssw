#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/GenEvent.h"

class HepMCCopy : public edm::one::EDProducer<> {
public:
  explicit HepMCCopy(edm::ParameterSet const& p);
  ~HepMCCopy() override = default;
  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
};

void HepMCCopy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", {"generatorSmeared"});
  descriptions.addWithDefaultLabel(desc);
}

HepMCCopy::HepMCCopy(edm::ParameterSet const& p)
    : hepMCToken_(consumes<edm::HepMCProduct>(p.getParameter<edm::InputTag>("src"))) {
  // This producer produces a HepMCProduct, a copy of the original one
  produces<edm::HepMCProduct>();
}

void HepMCCopy::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  const auto& theHepMCProduct = iEvent.getHandle(hepMCToken_);
  if (theHepMCProduct.isValid()) {
    auto pu_product = std::make_unique<edm::HepMCProduct>();
    iEvent.put(std::move(pu_product));
  } else {
    auto pu_product = std::make_unique<edm::HepMCProduct>(*theHepMCProduct);
    iEvent.put(std::move(pu_product));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HepMCCopy);
