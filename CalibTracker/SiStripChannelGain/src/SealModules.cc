
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibTracker/SiStripChannelGain/interface/SiStripGainRandomCalculator.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiStripGainRandomCalculator);
