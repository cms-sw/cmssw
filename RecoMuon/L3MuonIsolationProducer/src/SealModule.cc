#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_SEAL_MODULE();

#include "L3MuonIsolationProducer.h"
DEFINE_ANOTHER_FWK_MODULE(L3MuonIsolationProducer);

