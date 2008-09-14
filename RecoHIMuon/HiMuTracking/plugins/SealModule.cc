//#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;

#include "RecoHIMuon/HiMuTracking/plugins/TestMuL1L2Filter.h"
//#include "RecoHIMuon/HiMuTracking/plugins/TestMuL1L2FilterSTA.h"
using cms::TestMuL1L2Filter;
//using cms::TestMuL1L2FilterSTA;
DEFINE_ANOTHER_FWK_MODULE(TestMuL1L2Filter);
//DEFINE_ANOTHER_FWK_MODULE(TestMuL1L2FilterSTA);
