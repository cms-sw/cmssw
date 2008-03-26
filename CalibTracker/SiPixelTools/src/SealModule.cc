#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiPixelTools/interface/SiPixelOfflineCalibAnalysisBase.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelOfflineCalibAnalysisBase);

#include "CalibTracker/SiPixelTools/interface/SiPixelCalibDigiFilter.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelCalibDigiFilter);
