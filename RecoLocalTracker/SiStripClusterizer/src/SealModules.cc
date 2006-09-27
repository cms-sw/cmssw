
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfoProducer.h"

using cms::SiStripClusterizer;
using cms::SiStripClusterInfoProducer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterizer)
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterInfoProducer);
