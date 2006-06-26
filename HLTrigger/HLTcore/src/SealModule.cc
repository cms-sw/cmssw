#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "HLTrigger/HLTcore/interface/HLTAnalCand.h"
#include "HLTrigger/HLTcore/interface/HLTProdCand.h"
#include "HLTrigger/HLTcore/interface/HLTMakePathObject.h"
#include "HLTrigger/HLTcore/interface/HLTMakeGlobalObject.h"
#include "HLTrigger/HLTcore/interface/HLTMakeSummaryObjects.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTPrescaler)
DEFINE_ANOTHER_FWK_MODULE(HLTAnalCand)
DEFINE_ANOTHER_FWK_MODULE(HLTProdCand)
DEFINE_ANOTHER_FWK_MODULE(HLTMakePathObject)
DEFINE_ANOTHER_FWK_MODULE(HLTMakeGlobalObject)
DEFINE_ANOTHER_FWK_MODULE(HLTMakeSummaryObjects)
