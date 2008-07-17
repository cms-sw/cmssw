// $Id: SealModule.cc,v 1.3 2008/07/17 18:38:51 bachtis Exp $

#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgammaHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"
#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"
#include "DQMOffline/Trigger/interface/HLTTauCaloDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauTrkDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauElDQMOfflineSource.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FourVectorHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgammaHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgHLTOfflineClient);
DEFINE_ANOTHER_FWK_MODULE(HLTTauRefProducer);
DEFINE_ANOTHER_FWK_MODULE(HLTTauCaloDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauTrkDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauElDQMOfflineSource);



