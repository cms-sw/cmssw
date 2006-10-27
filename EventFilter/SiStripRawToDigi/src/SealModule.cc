#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiSource.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripDigiToRawModule);
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToDigiModule);
DEFINE_ANOTHER_FWK_MODULE(SiStripTrivialDigiSource);

