#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTHWFileReader.h"
#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"


  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(L1MuGMTHWFileReader);
  DEFINE_ANOTHER_FWK_MODULE(L1MuGlobalMuonTrigger);
