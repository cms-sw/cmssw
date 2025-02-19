#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripDCS/test/plugins/testbuilding.h"
DEFINE_ANOTHER_FWK_MODULE(testbuilding);

#include "CalibTracker/SiStripDCS/test/plugins/dpLocationMap.h"
DEFINE_ANOTHER_FWK_MODULE(dpLocationMap);
