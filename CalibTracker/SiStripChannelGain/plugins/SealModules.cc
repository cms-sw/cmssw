#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/ModuleFactory.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainRandomCalculator.h"
//#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainTickMarkCalculator.h"
#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainCosmicCalculator.h"
#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainFromAsciiFile.h"
#include "CalibTracker/SiStripChannelGain/interface/SiStripGainsPCLWorker.h"
#include "CalibTracker/SiStripChannelGain/interface/SiStripGainsPCLHarvester.h"

DEFINE_FWK_MODULE(SiStripGainRandomCalculator);
//DEFINE_FWK_MODULE(SiStripGainTickMarkCalculator);
DEFINE_FWK_MODULE(SiStripGainCosmicCalculator);
DEFINE_FWK_MODULE(SiStripGainFromAsciiFile);
DEFINE_FWK_MODULE(SiStripGainsPCLWorker);
DEFINE_FWK_MODULE(SiStripGainsPCLHarvester);
