
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverter.h"

using cms::SiStripRecHitConverter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripRecHitConverter)

