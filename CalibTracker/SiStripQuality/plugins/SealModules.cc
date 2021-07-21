
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripQualityHotStripIdentifier.h"
#include "CalibTracker/SiStripQuality/plugins/SiStripBadStripFromASCIIFile.h"

DEFINE_FWK_MODULE(SiStripQualityStatistics);
DEFINE_FWK_MODULE(SiStripQualityHotStripIdentifier);
DEFINE_FWK_MODULE(SiStripBadStripFromASCIIFile);
