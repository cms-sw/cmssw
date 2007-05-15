
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/GctRawToDigi/src/GctRawToDigi.h"
#include "EventFilter/GctRawToDigi/src/GctVmeToRaw.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GctVmeToRaw);
DEFINE_ANOTHER_FWK_MODULE(GctRawToDigi);
