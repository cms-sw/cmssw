
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Type1MET.h"
#include "MuonMET.h"
#include "TauMET.h"

using cms::Type1MET;
using cms::MuonMET;
using cms::TauMET;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(Type1MET);
DEFINE_ANOTHER_FWK_MODULE(MuonMET);
DEFINE_ANOTHER_FWK_MODULE(TauMET);
