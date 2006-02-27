#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMET/METProducers/interface/TowerMETProducer.h"

using cms::TowerMETProducer;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(TowerMETProducer)
