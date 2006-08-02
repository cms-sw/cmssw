
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/ClusterMTCCFilter.h"

using cms::MTCCHLTrigger;
using cms::ClusterMTCCFilter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MTCCHLTrigger)
DEFINE_ANOTHER_FWK_MODULE(ClusterMTCCFilter)
