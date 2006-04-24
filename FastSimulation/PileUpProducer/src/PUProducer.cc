#include "PluginManager/PluginManager.h"
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/src/VectorInputSourceFactory.h"

#include "FastSimulation/PileUpProducer/interface/HepMCProductContainer.h"
#include "FastSimulation/PileUpProducer/interface/PUProducer.h"

#include <iostream>
#include <memory>

PUProducer::PUProducer(edm::ParameterSet const & p) :
  input(edm::VectorInputSourceFactory::get()->makeVectorInputSource(p, edm::InputSourceDescription()).release())
{    
    produces<edm::HepMCProductContainer>();
}

PUProducer::~PUProducer() 
{ 
}

void PUProducer::beginJob(const edm::EventSetup & es)
{
    std::cout << " PUProducer initializing " << std::endl;
}
 
void PUProducer::endJob()
{ 
    std::cout << " PUProducer terminating " << std::endl; 
}
 
void PUProducer::produce(edm::Event & iEvent, const edm::EventSetup & es)
{

   std::auto_ptr<edm::HepMCProductContainer> pu(new edm::HepMCProductContainer);

   iEvent.put(pu);

}

DEFINE_FWK_MODULE(PUProducer)
