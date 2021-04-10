#ifndef RecoHGCal_TICL_PatternRecognitionPluginFactory_H
#define RecoHGCal_TICL_PatternRecognitionPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/plugins/GlobalCache.h"

typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<TICLLayerTiles>*(const edm::ParameterSet&,
                                                                                    const ticl::CacheBase*)>
    PatternRecognitionFactory;
typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>*(const edm::ParameterSet&,
                                                                                          const ticl::CacheBase*)>
    PatternRecognitionHFNoseFactory;

#endif
