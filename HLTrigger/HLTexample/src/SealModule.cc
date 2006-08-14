#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTexample/interface/HLTFiltCand.h"
#include "HLTrigger/HLTexample/interface/HLTSimpleJet.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTFiltCand)
DEFINE_ANOTHER_FWK_MODULE(HLTSimpleJet)
