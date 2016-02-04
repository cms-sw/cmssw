#include "FWCore/Framework/interface/MakerMacros.h"


#include "EventFilter/SiStripRawToDigi/plugins/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"

typedef sistrip::DigiToRawModule SiStripDigiToRawModule;
typedef sistrip::RawToDigiModule SiStripRawToDigiModule;
typedef sistrip::RawToClusters SiStripRawToClusters;

DEFINE_FWK_MODULE(SiStripDigiToRawModule);
DEFINE_FWK_MODULE(SiStripRawToDigiModule);
DEFINE_FWK_MODULE(SiStripRawToClusters);

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
DEFINE_FWK_MODULE(SiStripRawToClustersRoI);
