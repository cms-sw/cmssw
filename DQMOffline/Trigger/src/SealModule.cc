
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
#include "DQMOffline/Trigger/interface/BTagHLTOfflineSource.h"
#include "DQMOffline/Trigger/interface/BTagHLTOfflineClient.h"
#include "DQMOffline/Trigger/interface/DQMOfflineHLTEventInfoClient.h"
#include "DQMOffline/Trigger/interface/HLTTauCertifier.h"
#include "DQMOffline/Trigger/interface/TopHLTDiMuonDQM.h"
#include "DQMOffline/Trigger/interface/TopHLTDiMuonDQMClient.h"
#include "DQMOffline/Trigger/interface/HLTInclusiveVBFSource.h"
#include "DQMOffline/Trigger/interface/HLTInclusiveVBFClient.h"
#include "DQMOffline/Trigger/interface/TopDiLeptonHLTOfflineDQM.h"
#include "DQMOffline/Trigger/interface/TopSingleLeptonHLTOfflineDQM.h"

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FourVectorHLTClient);
DEFINE_FWK_MODULE(FourVectorHLTOffline);
DEFINE_FWK_MODULE(EgHLTOfflineSource);
DEFINE_FWK_MODULE(EgHLTOfflineClient);
DEFINE_FWK_MODULE(EgHLTOfflineSummaryClient);
DEFINE_FWK_MODULE(HLTTauRefProducer);
DEFINE_FWK_MODULE(HLTTauDQMOfflineSource);
DEFINE_FWK_MODULE(HLTTauPostProcessor);
DEFINE_FWK_MODULE(JetMETHLTOfflineSource);
DEFINE_FWK_MODULE(JetMETHLTOfflineClient);
DEFINE_FWK_MODULE(BTagHLTOfflineSource);
DEFINE_FWK_MODULE(BTagHLTOfflineClient);
DEFINE_FWK_MODULE(DQMOfflineHLTEventInfoClient);
DEFINE_FWK_MODULE(HLTTauCertifier);
DEFINE_FWK_MODULE(TopHLTDiMuonDQM);
DEFINE_FWK_MODULE(TopHLTDiMuonDQMClient);
DEFINE_FWK_MODULE(HLTInclusiveVBFSource);
DEFINE_FWK_MODULE(HLTInclusiveVBFClient);
DEFINE_FWK_MODULE(TopDiLeptonHLTOfflineDQM);
DEFINE_FWK_MODULE(TopSingleLeptonHLTOfflineDQM);
