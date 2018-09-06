#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/CTPPSRawToDigi/plugins/CTPPSPixelDigiToRaw.h"
#include "EventFilter/CTPPSRawToDigi/plugins/CTPPSTotemDigiToRaw.h"

DEFINE_FWK_MODULE(CTPPSPixelDigiToRaw);
DEFINE_FWK_MODULE(CTPPSTotemDigiToRaw);

