// $Id: SealModule.cc,v 1.1 2006/11/16 22:59:01 wittich Exp $
// 

#include "DQM/HLTEvF/interface/HltAnalyzer.h"
#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using edm::service::PathTimerService;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HltAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(PathTimerInserter);
DEFINE_ANOTHER_FWK_SERVICE(PathTimerService);
//DEFINE_ANOTHER_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);
 
