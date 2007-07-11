//#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibTracker/SiStripLorentzAngle/plugins/SiStripLAFakeESSource.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripLAFakeESSource);




