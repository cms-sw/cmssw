#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "DQM/HLTEvF/bin/HLTHistoryDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(HLTHistoryDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<HLTHistoryDQMService > > HLTDQMHistoryPopCon;
DEFINE_ANOTHER_FWK_MODULE(HLTDQMHistoryPopCon);
