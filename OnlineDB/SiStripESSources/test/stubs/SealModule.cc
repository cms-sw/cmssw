// Last commit: $Id: $
// Latest tag:  $Name:  $
// Location:    $Source: $

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PluginManager/ModuleDef.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripESSources/test/stubs/AnalyzeFedCabling.h"
DEFINE_ANOTHER_FWK_MODULE(AnalyzeFedCabling);

#include "OnlineDB/SiStripESSources/test/stubs/AnalyzePedestals.h"
DEFINE_ANOTHER_FWK_MODULE(AnalyzePedestals);

#include "OnlineDB/SiStripESSources/test/stubs/AnalyzeNoise.h"
DEFINE_ANOTHER_FWK_MODULE(AnalyzeNoise);

#include "OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilderFromDb.h"
DEFINE_ANOTHER_FWK_MODULE(test_FedCablingBuilderFromDb);
