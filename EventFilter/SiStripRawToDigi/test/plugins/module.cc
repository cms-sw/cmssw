#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "EventFilter/SiStripRawToDigi/test/plugins/AnalyzeSiStripDigis.h"
DEFINE_ANOTHER_FWK_MODULE(AnalyzeSiStripDigis);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialClusterSource.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripTrivialClusterSource);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialDigiSource.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripTrivialDigiSource);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripPerformanceRecordModule.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripPerformanceRecordModule);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersDummyRoI.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClustersDummyRoI);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersDummyUnpacker.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClustersDummyUnpacker);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClusters.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClusters);

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripClustersDSVBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripClustersDSVBuilder);

