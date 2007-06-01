#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SiPixelRawToDigi.h"
#include "SiPixelDigiToRaw.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelDigiToRaw);
DEFINE_ANOTHER_FWK_MODULE(SiPixelRawToDigi);
