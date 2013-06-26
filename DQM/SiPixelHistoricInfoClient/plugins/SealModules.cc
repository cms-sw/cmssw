#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"


#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelSummary.h"


//New approach

#include "DQM/SiPixelHistoricInfoClient/plugins/SiPixelHistoryDQMService.h"
DEFINE_FWK_SERVICE(SiPixelHistoryDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<SiPixelHistoryDQMService > > SiPixelDQMHistoryPopCon;
DEFINE_FWK_MODULE(SiPixelDQMHistoryPopCon);
