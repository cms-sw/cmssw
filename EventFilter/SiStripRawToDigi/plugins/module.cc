#include "FWCore/Framework/interface/MakerMacros.h"


#include "EventFilter/SiStripRawToDigi/plugins/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"
#include "EventFilter/SiStripRawToDigi/plugins/ExcludedFEDListProducer.h"

typedef sistrip::DigiToRawModule SiStripDigiToRawModule;
typedef sistrip::RawToDigiModule SiStripRawToDigiModule;
typedef sistrip::RawToClusters SiStripRawToClusters;
typedef sistrip::ExcludedFEDListProducer SiStripExcludedFEDListProducer;

DEFINE_FWK_MODULE(SiStripDigiToRawModule);
DEFINE_FWK_MODULE(SiStripRawToDigiModule);
DEFINE_FWK_MODULE(SiStripRawToClusters);
DEFINE_FWK_MODULE(SiStripExcludedFEDListProducer);

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
DEFINE_FWK_MODULE(SiStripRawToClustersRoI);
