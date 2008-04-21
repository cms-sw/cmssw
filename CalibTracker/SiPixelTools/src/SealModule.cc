#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiPixelTools/interface/SiPixelOfflineCalibAnalysisBase.h"
#include "CalibTracker/SiPixelTools/interface/SiPixelErrorsDigisToCalibDigis.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelOfflineCalibAnalysisBase);
DEFINE_ANOTHER_FWK_MODULE(SiPixelErrorsDigisToCalibDigis);
