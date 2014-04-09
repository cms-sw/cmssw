#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"



#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripFEDRawDataAnalyzer.h"
DEFINE_FWK_MODULE(SiStripFEDRawDataAnalyzer);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiAnalyzer.h"
DEFINE_FWK_MODULE(SiStripDigiAnalyzer);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialClusterSource.h"
DEFINE_FWK_MODULE(SiStripTrivialClusterSource);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialDigiSource.h"
DEFINE_FWK_MODULE(SiStripTrivialDigiSource);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiValidator.h"
DEFINE_FWK_MODULE(SiStripDigiValidator);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripClusterValidator.h"
DEFINE_FWK_MODULE(SiStripClusterValidator);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripModuleTimer.h"
DEFINE_FWK_MODULE(SiStripModuleTimer);
