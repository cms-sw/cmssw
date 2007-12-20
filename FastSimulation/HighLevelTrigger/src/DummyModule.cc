#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FastSimulation/HighLevelTrigger/interface/DummyModule.h"

DummyModule::DummyModule(edm::ParameterSet const & p) {

  produces<bool>();

}

DummyModule::~DummyModule() {}

void 
DummyModule::produce(edm::Event & e, const edm::EventSetup & c) {;}


DEFINE_FWK_MODULE(DummyModule);

