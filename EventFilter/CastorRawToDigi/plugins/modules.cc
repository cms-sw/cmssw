#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/CastorRawToDigi/plugins/CastorRawToDigi.h"
#include "EventFilter/CastorRawToDigi/plugins/CastorDigiToRaw.h"


DEFINE_FWK_MODULE(CastorRawToDigi);
DEFINE_FWK_MODULE(CastorDigiToRaw);
