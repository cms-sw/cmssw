#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibTracker/SiPixelTools/interface/SiPixelCalibDigiFilter.h"
#include "CalibTracker/SiPixelTools/interface/SiPixelErrorsDigisToCalibDigis.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiPixelCalibDigiFilter);
DEFINE_ANOTHER_FWK_MODULE(SiPixelErrorsDigisToCalibDigis);

