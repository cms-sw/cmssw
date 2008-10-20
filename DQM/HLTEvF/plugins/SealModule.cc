// $Id: SealModule.cc,v 1.15 2008/10/16 17:00:51 hdyoo Exp $

#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DQM/HLTEvF/interface/HLTMonElectronConsumer.h"
#include "DQM/HLTEvF/interface/FourVectorHLTOnline.h"
#include "DQM/HLTEvF/interface/HLTMon.h"
#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"
#include "DQM/HLTEvF/interface/HLTMonMuonClient.h"

using edm::service::PathTimerService;

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(PathTimerService);
//DEFINE_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PathTimerInserter);
DEFINE_FWK_MODULE(HLTTauDQMSource);
DEFINE_FWK_MODULE(HLTMuonDQMSource);
DEFINE_FWK_MODULE(HLTMonElectron);
DEFINE_FWK_MODULE(HLTMonElectronConsumer);
DEFINE_FWK_MODULE(HLTEventInfoClient);
DEFINE_FWK_MODULE(FourVectorHLTOnline);
DEFINE_FWK_MODULE(HLTMon);
DEFINE_FWK_MODULE(HLTMonMuonClient);

