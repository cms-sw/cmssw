// $Id: SealModule.cc,v 1.21 2009/11/14 00:13:22 rekovic Exp $

#include "DQMOffline/Trigger/interface/FourVectorHLTClient.h"
#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"
#include "DQMOffline/Trigger/interface/EgHLTOfflineSource.h"
#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"
#include "DQMOffline/Trigger/interface/EgHLTOfflineSummaryClient.h"
#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"
#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"
#include "DQMOffline/Trigger/interface/JetMETHLTOfflineSource.h"
#include "DQMOffline/Trigger/interface/JetMETHLTOfflineClient.h"
#include "DQMOffline/Trigger/interface/DQMOfflineHLTEventInfoClient.h"
#include "DQMOffline/Trigger/interface/TopElectronHLTOfflineSource.h"
#include "DQMOffline/Trigger/interface/TopElectronHLTOfflineClient.h"

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FourVectorHLTClient);
DEFINE_ANOTHER_FWK_MODULE(FourVectorHLTOffline);
DEFINE_ANOTHER_FWK_MODULE(EgHLTOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(EgHLTOfflineClient);
DEFINE_ANOTHER_FWK_MODULE(EgHLTOfflineSummaryClient);
DEFINE_ANOTHER_FWK_MODULE(HLTTauRefProducer);
DEFINE_ANOTHER_FWK_MODULE(HLTTauDQMOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(HLTTauPostProcessor);
DEFINE_ANOTHER_FWK_MODULE(JetMETHLTOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(JetMETHLTOfflineClient);
DEFINE_ANOTHER_FWK_MODULE(DQMOfflineHLTEventInfoClient);
DEFINE_ANOTHER_FWK_MODULE(TopElectronHLTOfflineSource);
DEFINE_ANOTHER_FWK_MODULE(TopElectronHLTOfflineClient);

