
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripClusterizer/test/TestCluster.h"

using cms::TestCluster;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestCluster);

