#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/special/interface/HLTPixlMBFilt.h"
#include "HLTrigger/special/interface/HLTPixlMBSelectFilter.h"
#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"
#include "HLTrigger/special/interface/HLTEcalIsolationFilter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTPixlMBFilt);
DEFINE_ANOTHER_FWK_MODULE(HLTPixlMBSelectFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTPixelIsolTrackFilter);				   
DEFINE_ANOTHER_FWK_MODULE(HLTEcalIsolationFilter);				   

