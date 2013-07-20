// $Id: SealModule.cc,v 1.29 2012/03/09 15:01:45 halil Exp $

#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "DQM/HLTEvF/interface/HLTMonBitSummary.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DQM/HLTEvF/interface/HLTMonElectronConsumer.h"
#include "DQM/HLTEvF/interface/FourVectorHLTOnline.h"
#include "DQM/HLTEvF/interface/HLTMon.h"
#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"
#include "DQM/HLTEvF/interface/HLTAlCaMonPi0.h"
#include "DQM/HLTEvF/interface/HLTAlCaMonEcalPhiSym.h"
#include "DQM/HLTEvF/interface/HLTOniaSource.h"
//#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonMuonClient.h"
#include "DQM/HLTEvF/interface/HLTJetMETDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonSimpleBTag.h"
#include "DQM/HLTEvF/interface/TrigResRateMon.h"

using edm::service::PathTimerService;

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(PathTimerService);
//DEFINE_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PathTimerInserter);
DEFINE_FWK_MODULE(HLTMonBitSummary);
DEFINE_FWK_MODULE(HLTMonElectron);
//DEFINE_FWK_MODULE(HLTMuonDQMSource);
DEFINE_FWK_MODULE(HLTMonMuonClient);
DEFINE_FWK_MODULE(HLTJetMETDQMSource);
DEFINE_FWK_MODULE(HLTMonElectronConsumer);
DEFINE_FWK_MODULE(HLTEventInfoClient);
DEFINE_FWK_MODULE(FourVectorHLTOnline);
DEFINE_FWK_MODULE(HLTMon);
DEFINE_FWK_MODULE(HLTAlCaMonPi0);
DEFINE_FWK_MODULE(HLTAlCaMonEcalPhiSym);
DEFINE_FWK_MODULE(HLTOniaSource);
DEFINE_FWK_MODULE(TrigResRateMon);
DEFINE_FWK_MODULE(HLTMonSimpleBTag);
