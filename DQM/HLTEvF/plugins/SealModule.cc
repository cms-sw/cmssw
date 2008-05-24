// $Id: SealModule.cc,v 1.4 2008/05/13 13:51:39 fwyzard Exp $

#include "DQM/HLTEvF/interface/HltAnalyzer.h"
#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DQM/HLTEvF/interface/FourVectorHLT.h"

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

