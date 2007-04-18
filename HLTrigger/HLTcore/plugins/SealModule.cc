#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "HLTrigger/HLTcore/interface/HLTMakeSummaryObjects.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTPrescaler);
DEFINE_ANOTHER_FWK_MODULE(HLTMakeSummaryObjects);
