#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/L1TMonitorClient/interface/L1TDTTFClient.h>
DEFINE_FWK_MODULE(L1TDTTFClient);

#include <DQM/L1TMonitorClient/interface/L1TDTTPGClient.h>
DEFINE_FWK_MODULE(L1TDTTPGClient);

#include "DQM/L1TMonitorClient/interface/L1TRPCTFClient.h"
DEFINE_FWK_MODULE(L1TRPCTFClient);

#include "DQM/L1TMonitorClient/interface/L1TdeCSCTPGClient.h"
DEFINE_FWK_MODULE(L1TdeCSCTPGClient);

#include "DQM/L1TMonitorClient/interface/L1TCSCTFClient.h"
DEFINE_FWK_MODULE(L1TCSCTFClient);

#include <DQM/L1TMonitorClient/interface/L1TGMTClient.h>
DEFINE_FWK_MODULE(L1TGMTClient);

#include <DQM/L1TMonitorClient/interface/L1TGCTClient.h>
DEFINE_FWK_MODULE(L1TGCTClient);

#include <DQM/L1TMonitorClient/interface/L1TEventInfoClient.h>
DEFINE_FWK_MODULE(L1TEventInfoClient);

#include <DQM/L1TMonitorClient/interface/L1EmulatorErrorFlagClient.h>
DEFINE_FWK_MODULE(L1EmulatorErrorFlagClient);

#include <DQM/L1TMonitorClient/interface/L1TStage2CaloLayer2DEClient.h>
DEFINE_FWK_MODULE(L1TStage2CaloLayer2DEClient);

#include <DQM/L1TMonitorClient/interface/L1TStage2CaloLayer2DEClientSummary.h>
DEFINE_FWK_MODULE(L1TStage2CaloLayer2DEClientSummary);

#include <DQM/L1TMonitorClient/interface/L1TStage2RatioClient.h>
DEFINE_FWK_MODULE(L1TStage2RatioClient);

#include <DQM/L1TMonitorClient/interface/L1TEMTFEventInfoClient.h>
DEFINE_FWK_MODULE(L1TEMTFEventInfoClient);
