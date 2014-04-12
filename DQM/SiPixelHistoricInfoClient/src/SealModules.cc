#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoDQMClient.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoEDAClient.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoReader.h"




DEFINE_FWK_MODULE(SiPixelHistoricInfoDQMClient);
DEFINE_FWK_MODULE(SiPixelHistoricInfoEDAClient);
DEFINE_FWK_MODULE(SiPixelHistoricInfoReader);
