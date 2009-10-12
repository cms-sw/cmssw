#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "DQM/SiStripHistoricInfoClient/plugins/SiStripHistoryDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripHistoryDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<SiStripHistoryDQMService > > SiStripDQMHistoryPopCon;
DEFINE_ANOTHER_FWK_MODULE(SiStripDQMHistoryPopCon);
