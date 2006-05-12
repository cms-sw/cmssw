#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTcore/interface/HLTSimpleJet.h"
#include "HLTrigger/HLTcore/interface/HLTMakePathObject.h"
#include "HLTrigger/HLTcore/interface/HLTMakeGlobalObject.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTSimpleJet)
DEFINE_ANOTHER_FWK_MODULE(HLTMakePathObject)
DEFINE_ANOTHER_FWK_MODULE(HLTMakeGlobalObject)
