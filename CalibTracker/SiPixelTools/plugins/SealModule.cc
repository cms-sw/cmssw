#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiPixelTools/interface/SiPixelCalibDigiFilter.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelCalibDigiFilter);
