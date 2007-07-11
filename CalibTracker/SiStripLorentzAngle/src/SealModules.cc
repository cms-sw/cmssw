#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngle.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleDB.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLAProfileBooker.h"

DEFINE_SEAL_MODULE();
DEFINE_FWK_MODULE(SiStripLorentzAngle);
DEFINE_FWK_MODULE(SiStripLorentzAngleDB);
DEFINE_FWK_MODULE(SiStripLAProfileBooker);

