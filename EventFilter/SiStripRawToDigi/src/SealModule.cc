#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripDigiToRawModule);

#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToDigiModule);

#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiSource.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripTrivialDigiSource);

#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersModule.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClustersModule);

