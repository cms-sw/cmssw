#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/ModuleFactory.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainRandomCalculator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripGainRandomCalculator);


