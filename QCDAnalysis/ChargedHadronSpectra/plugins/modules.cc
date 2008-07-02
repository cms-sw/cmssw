#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

DEFINE_SEAL_MODULE();

// Producers
#include "EnergyLossProducer.h"
DEFINE_ANOTHER_FWK_MODULE(EnergyLossProducer);

