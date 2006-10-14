
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/ClusterMTCCFilter.h"
#include "EventFilter/SiStripChannelChargeFilter/interface/TrackMTCCFilter.h"

using cms::MTCCHLTrigger;
using cms::ClusterMTCCFilter;
using cms::TrackMTCCFilter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MTCCHLTrigger)
DEFINE_ANOTHER_FWK_MODULE(ClusterMTCCFilter)
DEFINE_ANOTHER_FWK_MODULE(TrackMTCCFilter)
