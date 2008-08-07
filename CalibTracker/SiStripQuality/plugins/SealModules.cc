
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripBadModuleByHandBuilder.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripQualityHotStripIdentifier.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripBadStripFromASCIIFile.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripBadModuleByHandBuilder);
DEFINE_ANOTHER_FWK_MODULE(SiStripQualityStatistics);
DEFINE_ANOTHER_FWK_MODULE(SiStripQualityHotStripIdentifier);
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripFromASCIIFile);
