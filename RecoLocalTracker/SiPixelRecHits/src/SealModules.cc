#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelRecHitConverter.h"

using cms::SiPixelRecHitConverter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelRecHitConverter)
