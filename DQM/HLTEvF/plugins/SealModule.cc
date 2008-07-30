// $Id: SealModule.cc,v 1.5.2.1 2008/06/11 11:54:51 lorenzo Exp $

#include "DQM/HLTEvF/interface/HltAnalyzer.h"
#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DQM/HLTEvF/interface/FourVectorHLT.h"
#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"

using edm::service::PathTimerService;

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(PathTimerService);
//DEFINE_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PathTimerInserter);
DEFINE_FWK_MODULE(HltAnalyzer);
DEFINE_FWK_MODULE(HLTMuonDQMSource);
DEFINE_FWK_MODULE(HLTMonElectron);
DEFINE_FWK_MODULE(FourVectorHLT);
DEFINE_FWK_MODULE(HLTEventInfoClient);

