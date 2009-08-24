#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripLorentzAngle/plugins/EnsembleCalibrationLA.h"
DEFINE_ANOTHER_FWK_MODULE(sistrip::EnsembleCalibrationLA);

#include "CalibTracker/SiStripLorentzAngle/plugins/MeasureLA.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(sistrip::MeasureLA);
