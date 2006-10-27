#include "FWCore/Framework/interface/MakerMacros.h"
#include "PluginManager/ModuleDef.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripESSources/test/stubs/SiStripPopulateConfigDb.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripPopulateConfigDb);

#include "OnlineDB/SiStripESSources/test/stubs/SiStripAnalyzeFedCabling.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripAnalyzeFedCabling);

#include "OnlineDB/SiStripESSources/test/stubs/TestFedCablingBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(TestFedCablingBuilder);
