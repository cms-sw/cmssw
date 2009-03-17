#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizer.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterizer);

#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterProducer.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterProducer);
