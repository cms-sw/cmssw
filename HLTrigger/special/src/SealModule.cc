#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/special/interface/HLTPixlMBFilt.h"
#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTPixlMBFilt);
DEFINE_ANOTHER_FWK_MODULE(HLTPixelIsolTrackFilter);				   
