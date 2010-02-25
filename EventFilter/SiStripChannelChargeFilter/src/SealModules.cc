
#include "FWCore/PluginManager/interface/ModuleDef.h"

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

DEFINE_FWK_MODULE(MTCCHLTrigger);
DEFINE_FWK_MODULE(ClusterMTCCFilter);
DEFINE_FWK_MODULE(TECClusterFilter);
DEFINE_FWK_MODULE(TrackMTCCFilter);
DEFINE_FWK_MODULE(LTCTriggerBitsFilter);
