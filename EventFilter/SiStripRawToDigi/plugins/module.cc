#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "EventFilter/SiStripRawToDigi/plugins/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"

DEFINE_ANOTHER_FWK_MODULE(OldSiStripDigiToRawModule);
DEFINE_ANOTHER_FWK_MODULE(OldSiStripRawToDigiModule);
DEFINE_ANOTHER_FWK_MODULE(OldSiStripRawToClusters);

typedef sistrip::DigiToRawModule SiStripDigiToRawModule;
typedef sistrip::RawToDigiModule SiStripRawToDigiModule;
typedef sistrip::RawToClusters SiStripRawToClusters;

DEFINE_ANOTHER_FWK_MODULE(SiStripDigiToRawModule);
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToDigiModule);
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClusters);

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClustersRoI);
