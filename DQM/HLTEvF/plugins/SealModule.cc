// $Id: SealModule.cc,v 1.18 2009/06/17 01:03:52 hdyoo Exp $

#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonBitSummary.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DQM/HLTEvF/interface/HLTMonElectronConsumer.h"
#include "DQM/HLTEvF/interface/FourVectorHLTOnline.h"
#include "DQM/HLTEvF/interface/HLTMon.h"
#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"
#include "DQM/HLTEvF/interface/HLTMonMuonClient.h"
#include "DQM/HLTEvF/interface/HLTMonJetMETConsumer.h"
#include "DQM/HLTEvF/interface/HLTMonJetMETDQMSource.h"

using edm::service::PathTimerService;

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(PathTimerService);
//DEFINE_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PathTimerInserter);
DEFINE_FWK_MODULE(HLTTauDQMSource);
DEFINE_FWK_MODULE(HLTMuonDQMSource);
DEFINE_FWK_MODULE(HLTMonBitSummary);
DEFINE_FWK_MODULE(HLTMonElectron);
DEFINE_FWK_MODULE(HLTMonElectronConsumer);
DEFINE_FWK_MODULE(HLTEventInfoClient);
DEFINE_FWK_MODULE(FourVectorHLTOnline);
DEFINE_FWK_MODULE(HLTMon);
DEFINE_FWK_MODULE(HLTMonMuonClient);
DEFINE_FWK_MODULE(HLTMonJetMETConsumer);
DEFINE_FWK_MODULE(HLTMonJetMETDQMSource);
