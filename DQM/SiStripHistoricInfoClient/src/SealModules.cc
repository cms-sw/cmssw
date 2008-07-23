#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/SiStripHistoricInfoClient/interface/HistoricOfflineClient.h"
#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricPlot.h"
#include "DQM/SiStripHistoricInfoClient/interface/ReadFromFile.h"

using cms::SiStripHistoricPlot;

DEFINE_SEAL_MODULE();
DEFINE_FWK_MODULE(HistoricOfflineClient);
DEFINE_FWK_MODULE(SiStripHistoricPlot);
DEFINE_FWK_MODULE(ReadFromFile);
