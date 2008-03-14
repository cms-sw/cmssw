
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Type1MET.h"
#include "MuonMET.h"

using cms::Type1MET;
using cms::MuonMET;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(Type1MET);
DEFINE_ANOTHER_FWK_MODULE(MuonMET);
