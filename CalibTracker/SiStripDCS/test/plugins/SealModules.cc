#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripDCS/test/plugins/testSiStripDcuDetIdMap.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripDcuDetIdMap);

#include "CalibTracker/SiStripDCS/test/plugins/testbuilding.h"
DEFINE_ANOTHER_FWK_MODULE(testbuilding);

