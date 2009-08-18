#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripLorentzAngle/interface/plugins/LA_Calibration.h"
DEFINE_ANOTHER_FWK_MODULE(LA_Calibration);

#include "CalibTracker/SiStripLorentzAngle/interface/plugins/LA_Measurement.h"
DEFINE_ANOTHER_FWK_MODULE(LA_Measurement);

