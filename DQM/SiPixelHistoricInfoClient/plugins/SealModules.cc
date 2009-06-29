#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"


#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelSummary.h"


//New approach

#include "DQM/SiPixelHistoricInfoClient/plugins/SiPixelHistoryDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(SiPixelHistoryDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<SiPixelHistoryDQMService > > SiPixelDQMHistoryPopCon;
DEFINE_ANOTHER_FWK_MODULE(SiPixelDQMHistoryPopCon);
