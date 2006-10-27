#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"


  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_MODULE(L1GlobalTrigger);
