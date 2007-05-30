#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngle.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleDB.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripLorentzAngle);
DEFINE_ANOTHER_FWK_MODULE(SiStripLorentzAngleDB);
