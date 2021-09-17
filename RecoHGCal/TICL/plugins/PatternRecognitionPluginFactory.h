#ifndef RecoHGCal_TICL_PatternRecognitionPluginFactory_H
#define RecoHGCal_TICL_PatternRecognitionPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/interface/GlobalCache.h"

typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<TICLLayerTiles>*(
    const edm::ParameterSet&, const ticl::CacheBase*, edm::ConsumesCollector)>
    PatternRecognitionFactory;
typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>*(
    const edm::ParameterSet&, const ticl::CacheBase*, edm::ConsumesCollector)>
    PatternRecognitionHFNoseFactory;

#endif
