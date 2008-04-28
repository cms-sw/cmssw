#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQM/L1TMonitorClient/interface/L1THcalClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1THcalClient);
#include <DQM/L1TMonitorClient/interface/L1TDTTPGClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TDTTPGClient);
#include <DQM/L1TMonitorClient/interface/L1TdeECALClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TdeECALClient);
#include "DQM/L1TMonitorClient/interface/L1TRPCTFClient.h"
DEFINE_ANOTHER_FWK_MODULE(L1TRPCTFClient);
#include <DQM/L1TMonitorClient/interface/L1TCSCTFClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TCSCTFClient);
#include <DQM/L1TMonitorClient/interface/L1TGMTClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TGMTClient);
#include <DQM/L1TMonitorClient/interface/L1TGCTClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TGCTClient);
