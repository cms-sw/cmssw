#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;
  class HepMCProduct;
}  // namespace edm

class GeneratorSmearedProducer : public edm::global::EDProducer<> {
public:
  explicit GeneratorSmearedProducer(edm::ParameterSet const& p);

  void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<edm::HepMCProduct> newToken_;
  const edm::EDGetTokenT<edm::HepMCProduct> oldToken_;
};

GeneratorSmearedProducer::GeneratorSmearedProducer(edm::ParameterSet const& ps)
    : newToken_(consumes<edm::HepMCProduct>(ps.getUntrackedParameter<edm::InputTag>("currentTag"))),
      oldToken_(consumes<edm::HepMCProduct>(ps.getUntrackedParameter<edm::InputTag>("previousTag"))) {
  // This producer produces a HepMCProduct, a copy of the original one
  // It is used for backwaerd compatibility
  produces<edm::HepMCProduct>();
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
}

void GeneratorSmearedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("currentTag", edm::InputTag("VtxSmeared"));
  desc.addUntracked<edm::InputTag>("previousTag", edm::InputTag("generator"));
  descriptions.add("generatorSmeared", desc);
}

DEFINE_FWK_MODULE(GeneratorSmearedProducer);
