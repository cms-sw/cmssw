// $Id: SealModule.cc,v 1.8 2008/06/11 10:03:57 lorenzo Exp $


#include "DQM/HLTEvF/interface/HltAnalyzer.h"
#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQM/HLTEvF/interface/HLTTauDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DQM/HLTEvF/interface/FourVectorHLT.h"
#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"
#include "DQM/HLTEvF/interface/HLTMuonL1DQMSource.h"
#include "DQM/HLTEvF/interface/HLTMuonRecoDQMSource.h"
#include "DQM/HLTEvF/interface/HLTMuonIsoDQMSource.h"


using edm::service::PathTimerService;

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(PathTimerService);
DEFINE_FWK_SERVICE_MAKER(PathTimerService,PathTimerServiceMaker);

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PathTimerInserter);
DEFINE_FWK_MODULE(HltAnalyzer);
DEFINE_FWK_MODULE(HLTMuonDQMSource);
DEFINE_FWK_MODULE(HLTTauDQMSource);
DEFINE_FWK_MODULE(HLTMonElectron);
DEFINE_FWK_MODULE(FourVectorHLT);
DEFINE_FWK_MODULE(HLTEventInfoClient);
DEFINE_FWK_MODULE(HLTMuonL1DQMSource);
DEFINE_FWK_MODULE(HLTMuonRecoDQMSource);
DEFINE_FWK_MODULE(HLTMuonIsoDQMSource);


