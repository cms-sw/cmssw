#include "PluginManager/ModuleDef.h"
DEFINE_SEAL_MODULE();

#include "FWCore/Framework/interface/SourceFactory.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripFedCablingBuilderFromDb);
