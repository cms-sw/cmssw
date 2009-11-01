#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripCalibLorentzAngle.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLAProfileBooker.h"

DEFINE_SEAL_MODULE();
DEFINE_FWK_MODULE(SiStripCalibLorentzAngle);
DEFINE_FWK_MODULE(SiStripLAProfileBooker);

