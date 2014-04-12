#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "DQM/SiStripHistoricInfoClient/plugins/SiStripHistoryDQMService.h"
DEFINE_FWK_SERVICE(SiStripHistoryDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<SiStripHistoryDQMService > > SiStripDQMHistoryPopCon;
DEFINE_FWK_MODULE(SiStripDQMHistoryPopCon);
