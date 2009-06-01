#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoEDAClient.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoReader.h"


using cms::SiPixelHistoricInfoReader;


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiPixelHistoricInfoEDAClient);
DEFINE_ANOTHER_FWK_MODULE(SiPixelHistoricInfoReader);
