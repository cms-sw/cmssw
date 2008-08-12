#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

/*
****** TO BE REMOVED *******
#include "DQM/SiStripHistoricInfoClient/interface/HistoricOfflineClient.h"
#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricPlot.h"
#include "DQM/SiStripHistoricInfoClient/interface/ReadFromFile.h"
using cms::SiStripHistoricPlot;
DEFINE_FWK_MODULE(HistoricOfflineClient);
DEFINE_FWK_MODULE(SiStripHistoricPlot);
DEFINE_FWK_MODULE(ReadFromFile);
********************************
*/

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DQM/SiStripHistoricInfoClient/plugins/SiStripHistoricDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripHistoricDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiStrip/interface/SiStripPopConDbObjHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
typedef popcon::PopConAnalyzer< popcon::SiStripPopConDbObjHandler< SiStripSummary, SiStripHistoricDQMService > > SiStripPopConHistoricDQM;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConHistoricDQM);
