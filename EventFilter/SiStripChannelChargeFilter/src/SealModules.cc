
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/ClusterMTCCFilter.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/TECClusterFilter.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/TrackMTCCFilter.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/LTCTriggerBitsFilter.h"

using cms::MTCCHLTrigger;
using cms::ClusterMTCCFilter;
using cms::TECClusterFilter;
using cms::TrackMTCCFilter;
using cms::LTCTriggerBitsFilter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MTCCHLTrigger);
DEFINE_ANOTHER_FWK_MODULE(ClusterMTCCFilter);
DEFINE_ANOTHER_FWK_MODULE(TECClusterFilter);
DEFINE_ANOTHER_FWK_MODULE(TrackMTCCFilter);
DEFINE_ANOTHER_FWK_MODULE(LTCTriggerBitsFilter);
