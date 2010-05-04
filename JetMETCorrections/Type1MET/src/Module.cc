
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetMETCorrections/Type1MET/interface/Type1MET.h"
#include "JetMETCorrections/Type1MET/interface/TauMET.h"

using cms::Type1MET;
using cms::TauMET;



DEFINE_FWK_MODULE(Type1MET);
DEFINE_FWK_MODULE(TauMET);
