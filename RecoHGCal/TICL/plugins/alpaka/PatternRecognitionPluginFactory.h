#ifndef RecoHGCal_TICL_PatternRecognitionPluginFactory_H
#define RecoHGCal_TICL_PatternRecognitionPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/interface/alpaka/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/interface/alpaka/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/interface/GlobalCache.h"

using PatternRecognitionFactoryAlpaka =
    ::edmplugin::PluginFactory<ALPAKA_ACCELERATOR_NAMESPACE::PatternRecognitionAlgoBase*(const edm::ParameterSet&)>;
// using PatternRecognitionHFNoseFactoryAlpaka =
//     edmplugin::PluginFactory<ALPAKA_ACCELERATOR_NAMESPACE::PatternRecognitionAlgoBase*(const edm::ParameterSet&)>;

#endif
