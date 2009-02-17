
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"

using cms::SiStripClusterizer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterizer);
