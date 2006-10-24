#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/SiPixelRawToDigi/interface/SiPixelRawToDigi.h"
#include "EventFilter/SiPixelRawToDigi/interface/SiPixelDigiToRaw.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelDigiToRaw);
DEFINE_ANOTHER_FWK_MODULE(SiPixelRawToDigi);
