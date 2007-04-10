// $Id: SealModule.cc,v 1.2 2006/12/15 06:07:56 dlange Exp $
// 

#include "DQM/HLTEvF/interface/HltAnalyzer.h"
#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using edm::service::PathTimerService;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HltAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(PathTimerInserter);
DEFINE_ANOTHER_FWK_SERVICE(PathTimerService);
//DEFINE_ANOTHER_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);
 
