#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/L1TMonitorClient/interface/L1TDTTFClient.h>
DEFINE_FWK_MODULE(L1TDTTFClient);

#include <DQM/L1TMonitorClient/interface/L1TDTTPGClient.h>
DEFINE_FWK_MODULE(L1TDTTPGClient);

#include "DQM/L1TMonitorClient/interface/L1TRPCTFClient.h"
DEFINE_FWK_MODULE(L1TRPCTFClient);

#include <DQM/L1TMonitorClient/interface/L1TCSCTFClient.h>
DEFINE_FWK_MODULE(L1TCSCTFClient);

#include <DQM/L1TMonitorClient/interface/L1TGMTClient.h>
DEFINE_FWK_MODULE(L1TGMTClient);

#include <DQM/L1TMonitorClient/interface/L1TGCTClient.h>
DEFINE_FWK_MODULE(L1TGCTClient);

#include <DQM/L1TMonitorClient/interface/L1TEventInfoClient.h>
DEFINE_FWK_MODULE(L1TEventInfoClient);

#include <DQM/L1TMonitorClient/interface/L1EmulatorErrorFlagClient.h>
DEFINE_FWK_MODULE(L1EmulatorErrorFlagClient);
