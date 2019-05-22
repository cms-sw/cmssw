#include "FWCore/Framework/interface/MakerMacros.h"

#include "OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilder.h"
DEFINE_FWK_MODULE(test_FedCablingBuilder);

#include "OnlineDB/SiStripESSources/test/stubs/test_AnalyzeCabling.h"
DEFINE_FWK_MODULE(test_AnalyzeCabling);

#include "OnlineDB/SiStripESSources/test/stubs/test_PedestalsBuilder.h"
DEFINE_FWK_MODULE(test_PedestalsBuilder);

#include "OnlineDB/SiStripESSources/test/stubs/test_NoiseBuilder.h"
DEFINE_FWK_MODULE(test_NoiseBuilder);

#include "OnlineDB/SiStripESSources/test/stubs/testSiStripGainBuilderFromDb.h"
DEFINE_FWK_MODULE(testSiStripGainBuilderFromDb);
