#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/MinBias/interface/MinBias.h"
using cms::MinBias;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MinBias);
