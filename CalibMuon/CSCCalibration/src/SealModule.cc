#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondFormats/CSCObjects/interface/CSCFakeConditionsProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeConditionsProducer);
