#include "GeneratorInterface/Core/interface/GeneratorSmearedProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <memory>

GeneratorSmearedProducer::GeneratorSmearedProducer(edm::ParameterSet const& ps) :
  newToken_(consumes<edm::HepMCProduct>(ps.getUntrackedParameter<edm::InputTag>("currentTag"))),
  oldToken_(consumes<edm::HepMCProduct>(ps.getUntrackedParameter<edm::InputTag>("previousTag"))) {

  // This producer produces a HepMCProduct, a copy of the original one
  // It is used for backwaerd compatibility
  produces<edm::HepMCProduct>();
}

void GeneratorSmearedProducer::produce(edm::Event & iEvent, const edm::EventSetup & es) {
  edm::Handle<edm::HepMCProduct> theHepMCProduct;
  bool found = iEvent.getByToken(newToken_,theHepMCProduct);
  if (!found) { 
    found = iEvent.getByToken(oldToken_,theHepMCProduct);
  } 
  if (found) { 
    std::unique_ptr<edm::HepMCProduct> theCopy(new edm::HepMCProduct(*theHepMCProduct));  
    iEvent.put(std::move(theCopy));
  }
}

void GeneratorSmearedProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   desc.addUntracked<edm::InputTag>("currentTag", edm::InputTag("VtxSmeared"));
   desc.addUntracked<edm::InputTag>("previousTag", edm::InputTag("generator"));
   descriptions.add("generatorSmeared", desc);
}
