#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESDataIntegrityClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESOccupancyClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalTBClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCTClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCMCTClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCMTBClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESTDCCTClient.h"
          
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ESDataIntegrityClient);
DEFINE_ANOTHER_FWK_MODULE(ESOccupancyClient);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalTBClient);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalCTClient);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalCMCTClient);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalCMTBClient);
DEFINE_ANOTHER_FWK_MODULE(ESTDCCTClient);

