// $Id: SealModule.cc,v 1.12 2008/10/29 14:49:08 rekovic Exp $

#include "DQMOffline/Trigger/interface/FourVectorHLTClient.h"
#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgammaHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"
#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"
#include "DQMOffline/Trigger/interface/HLTTauL1DQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauCaloDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauTrkDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauElDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FourVectorHLTClient);
DEFINE_FWK_MODULE(FourVectorHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgammaHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgHLTOfflineClient);
DEFINE_ANOTHER_FWK_MODULE(HLTTauRefProducer);
DEFINE_ANOTHER_FWK_MODULE(HLTTauL1DQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauCaloDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauTrkDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauElDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauPostProcessor);


