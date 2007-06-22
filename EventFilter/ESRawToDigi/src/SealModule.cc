//#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"
#include "EventFilter/ESRawToDigi/interface/ESRawToDigiTB.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ESRawToDigi);
DEFINE_ANOTHER_FWK_MODULE(ESRawToDigiTB);


