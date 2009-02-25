
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizer.h"

using cms::SiStripClusterizer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterizer);
