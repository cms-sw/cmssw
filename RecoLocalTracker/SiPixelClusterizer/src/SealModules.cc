
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterProducer.h"

using cms::SiPixelClusterProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelClusterProducer)

