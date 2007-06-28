#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "CalibTracker/SiPixelLorentzAngle/interface/SiPixelLorentzAngle.h"
#include "CalibTracker/SiPixelLorentzAngle/interface/SiPixelLorentzAngleDB.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelLorentzAngle);
DEFINE_ANOTHER_FWK_MODULE(SiPixelLorentzAngleDB);
