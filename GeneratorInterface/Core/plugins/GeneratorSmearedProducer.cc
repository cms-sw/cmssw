#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;
}  // namespace edm

class GeneratorSmearedProducer : public edm::global::EDProducer<> {
public:
  explicit GeneratorSmearedProducer(edm::ParameterSet const& p);

  void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<edm::HepMCProduct> newToken_;
  const edm::EDGetTokenT<edm::HepMCProduct> oldToken_;
  const edm::EDGetTokenT<edm::HepMC3Product> Token3_;
};

GeneratorSmearedProducer::GeneratorSmearedProducer(edm::ParameterSet const& ps)
    : newToken_(consumes<edm::HepMCProduct>(ps.getUntrackedParameter<edm::InputTag>("currentTag"))),
      oldToken_(consumes<edm::HepMCProduct>(ps.getUntrackedParameter<edm::InputTag>("previousTag"))),
      Token3_(consumes<edm::HepMC3Product>(ps.getUntrackedParameter<edm::InputTag>("currentTag"))) {
  // This producer produces a HepMCProduct, a copy of the original one
  // It is used for backward compatibility
  // If HepMC3Product exists, it produces its copy
  // It adds "generatorSmeared" to description, which is needed for further processing
  produces<edm::HepMCProduct>();
  produces<edm::HepMC3Product>();
}

void GeneratorSmearedProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  edm::Handle<edm::HepMCProduct> theHepMCProduct;
  bool found = iEvent.getByToken(newToken_, theHepMCProduct);
  if (!found) {
    found = iEvent.getByToken(oldToken_, theHepMCProduct);
  }
  if (found) {
    std::unique_ptr<edm::HepMCProduct> theCopy(new edm::HepMCProduct(*theHepMCProduct));
    iEvent.put(std::move(theCopy));
  }

  edm::Handle<edm::HepMC3Product> theHepMC3Product;
  found = iEvent.getByToken(Token3_, theHepMC3Product);
  if (found) {
    std::unique_ptr<edm::HepMC3Product> theCopy3(new edm::HepMC3Product(*theHepMC3Product));
    iEvent.put(std::move(theCopy3));
  }
}

void GeneratorSmearedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("currentTag", edm::InputTag("VtxSmeared"));
  desc.addUntracked<edm::InputTag>("previousTag", edm::InputTag("generator"));
  descriptions.add("generatorSmeared", desc);
}

DEFINE_FWK_MODULE(GeneratorSmearedProducer);
