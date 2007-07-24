#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTexample/interface/HLTAnalFilt.h"
#include "HLTrigger/HLTexample/interface/HLTProdCand.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTAnalFilt);
DEFINE_ANOTHER_FWK_MODULE(HLTProdCand);
