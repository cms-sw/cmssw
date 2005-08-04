#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMET/METProducers/interface/UncorrTowersMETProducer.h"

using cms::UncorrTowersMETProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(UncorrTowersMETProducer)
