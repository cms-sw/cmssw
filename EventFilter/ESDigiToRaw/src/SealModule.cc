//#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/ESDigiToRaw/interface/ESDigiToRaw.h"
#include "EventFilter/ESDigiToRaw/interface/ESDigiToRawTB.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ESDigiToRaw);
DEFINE_ANOTHER_FWK_MODULE(ESDigiToRawTB);


