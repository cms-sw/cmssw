#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "EventFilter/SiStripRawToDigi/plugins/SiStripDigiToRawModule.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripDigiToRawModule);

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToDigiModule);

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClustersRoI);

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripRawToClusters);

using namespace sistrip;
DEFINE_ANOTHER_FWK_MODULE(RawToDigiModule);
DEFINE_ANOTHER_FWK_MODULE(RawToClusters);


