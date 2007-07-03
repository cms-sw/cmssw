#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalTBTask.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalCTTask.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyTBTask.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESDataIntegrityTask.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalCMCTTask.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalCMTBTask.h"
          
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ESPedestalTBTask);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalCTTask);
DEFINE_ANOTHER_FWK_MODULE(ESOccupancyTBTask);
DEFINE_ANOTHER_FWK_MODULE(ESDataIntegrityTask);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalCMCTTask);
DEFINE_ANOTHER_FWK_MODULE(ESPedestalCMTBTask);
