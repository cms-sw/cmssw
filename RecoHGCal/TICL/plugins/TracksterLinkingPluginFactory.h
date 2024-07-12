#ifndef RecoHGCal_TICL_TracksterLinkingPluginFactory_H
#define RecoHGCal_TICL_TracksterLinkingPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"

using TracksterLinkingPluginFactory =
    edmplugin::PluginFactory<ticl::TracksterLinkingAlgoBase*(const edm::ParameterSet&, edm::ConsumesCollector)>;

#endif
