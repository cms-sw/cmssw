#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticlesProd.h"
#include "L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticleMapProd.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1ExtraParticlesProd);
DEFINE_ANOTHER_FWK_MODULE(L1ExtraParticleMapProd);
