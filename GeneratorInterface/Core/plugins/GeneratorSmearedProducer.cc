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
  edm::EDGetTokenT<edm::HepMCProduct> newToken_;
  edm::EDGetTokenT<edm::HepMCProduct> oldToken_;
  edm::EDGetTokenT<edm::HepMC3Product> Token3_;
};

namespace {
  template <typename T>
  bool match(edm::InputTag const& iTag, edm::ProductDescription const& iDesc) {
    if (iDesc.unwrappedTypeID() == edm::TypeID(typeid(T))) {
      if (iDesc.moduleLabel() == iTag.label() and iDesc.productInstanceName() == iTag.instance()) {
        if (iTag.process().empty() or iTag.willSkipCurrentProcess() or
            iTag.process() == edm::InputTag::kCurrentProcess) {
          return true;
        }
      } else {
        return iTag.process() == iDesc.processName();
      }
    }
    return false;
  }
}  // namespace

GeneratorSmearedProducer::GeneratorSmearedProducer(edm::ParameterSet const& ps) {
  // This producer produces a HepMCProduct, a copy of the original one
  // It is used for backward compatibility
  // If HepMC3Product exists, it produces its copy
  // It adds "generatorSmeared" to description, which is needed for further processing
  auto currentTag = ps.getUntrackedParameter<edm::InputTag>("currentTag");
  auto previousTag = ps.getUntrackedParameter<edm::InputTag>("previousTag");
  callWhenNewProductsRegistered([this, currentTag, previousTag](edm::ProductDescription const& desc) {
    bool oldHep = false;
    if (match<edm::HepMCProduct>(currentTag, desc)) {
      if (newToken_.isUninitialized()) {
        newToken_ = consumes<edm::HepMCProduct>(currentTag);
        oldHep = true;
      }
    }
    if (match<edm::HepMCProduct>(previousTag, desc)) {
      if (oldToken_.isUninitialized()) {
        oldToken_ = consumes<edm::HepMCProduct>(previousTag);
        oldHep = true;
      }
    }
    if (oldHep) {
      produces<edm::HepMCProduct>();
    }
    if (match<edm::HepMC3Product>(currentTag, desc)) {
      Token3_ = consumes<edm::HepMC3Product>(currentTag);
      produces<edm::HepMC3Product>();
    }
  });
}

void GeneratorSmearedProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  edm::Handle<edm::HepMCProduct> theHepMCProduct;
  bool found = false;
  if (not newToken_.isUninitialized()) {
    found = iEvent.getByToken(newToken_, theHepMCProduct);
  }
  if (!found and not oldToken_.isUninitialized()) {
    found = iEvent.getByToken(oldToken_, theHepMCProduct);
  }
  if (found) {
    std::unique_ptr<edm::HepMCProduct> theCopy(new edm::HepMCProduct(*theHepMCProduct));
    iEvent.put(std::move(theCopy));
  }

  if (not Token3_.isUninitialized()) {
    edm::Handle<edm::HepMC3Product> theHepMC3Product;
    found = iEvent.getByToken(Token3_, theHepMC3Product);
    if (found) {
      std::unique_ptr<edm::HepMC3Product> theCopy3(new edm::HepMC3Product(*theHepMC3Product));
      iEvent.put(std::move(theCopy3));
    }
  }
}

void GeneratorSmearedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("currentTag", edm::InputTag("VtxSmeared"));
  desc.addUntracked<edm::InputTag>("previousTag", edm::InputTag("generator"));
  descriptions.add("generatorSmeared", desc);
}

DEFINE_FWK_MODULE(GeneratorSmearedProducer);
