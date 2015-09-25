#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "RecoParticleFlow/Configuration/plugins/HepMCCopy.h"
#include "HepMC/GenEvent.h"

HepMCCopy::HepMCCopy(edm::ParameterSet const & p)  
{    
  // This producer produces a HepMCProduct, a copy of the original one
  produces<edm::HepMCProduct>();
}

void HepMCCopy::produce(edm::Event & iEvent, const edm::EventSetup & es)
{
  
  edm::Handle<edm::HepMCProduct> theHepMCProduct;
  bool source = iEvent.getByLabel("generatorSmeared",theHepMCProduct);
  if ( !source ) { 
    std::auto_ptr<edm::HepMCProduct> pu_product(new edm::HepMCProduct());  
    iEvent.put(pu_product);
  } else { 
    std::auto_ptr<edm::HepMCProduct> pu_product(new edm::HepMCProduct(*theHepMCProduct));  
    iEvent.put(pu_product);
  }

}

DEFINE_FWK_MODULE(HepMCCopy);
