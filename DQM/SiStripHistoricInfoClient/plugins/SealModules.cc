#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

// //SiStrip private (and original) approach
#include "DQM/SiStripHistoricInfoClient/plugins/SiStripHistoricDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripHistoricDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiStrip/interface/SiStripPopConDbObjHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
typedef popcon::PopConAnalyzer< popcon::SiStripPopConDbObjHandler< SiStripSummary, SiStripHistoricDQMService > > SiStripPopConHistoricDQM;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConHistoricDQM);

//---------------------------------------------------------------

//New approach

#include "DQM/SiStripHistoricInfoClient/plugins/SiStripHistoryDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripHistoryDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<SiStripHistoryDQMService > > SiStripDQMHistoryPopCon;
DEFINE_ANOTHER_FWK_MODULE(SiStripDQMHistoryPopCon);
