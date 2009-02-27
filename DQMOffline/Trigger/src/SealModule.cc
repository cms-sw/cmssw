// $Id: SealModule.cc,v 1.15 2009/02/24 19:59:39 dlange Exp $

#include "DQMOffline/Trigger/interface/FourVectorHLTClient.h"
#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgammaHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"

#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FourVectorHLTClient);
DEFINE_ANOTHER_FWK_MODULE(FourVectorHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgammaHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgHLTOfflineClient);
DEFINE_ANOTHER_FWK_MODULE(HLTTauRefProducer);
DEFINE_ANOTHER_FWK_MODULE(HLTTauDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauPostProcessor);


