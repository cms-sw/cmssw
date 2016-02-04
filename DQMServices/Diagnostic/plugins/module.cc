#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//*****DEFINE THE USER SPECIFIC HDQM SERVICE*****//

#include "DQMServices/Diagnostic/plugins/GenericHistoryDQMService.h"
DEFINE_FWK_SERVICE(GenericHistoryDQMService);


//***** DEFINE THE POPCON EDANALYZER *****//
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryPopConHandler.h"
typedef popcon::PopConAnalyzer< popcon::DQMHistoryPopConHandler<GenericHistoryDQMService > > GenericDQMHistoryPopCon;
DEFINE_FWK_MODULE(GenericDQMHistoryPopCon);
