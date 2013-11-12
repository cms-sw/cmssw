#ifndef PixelTriplets_HitTripletGeneratorFromPairAndLayersFactory_H 
#define PixelTriplets_HitTripletGeneratorFromPairAndLayersFactory_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet; class ConsumesCollector;}

typedef edmplugin::PluginFactory<HitTripletGeneratorFromPairAndLayers *(const edm::ParameterSet &, edm::ConsumesCollector&)>
	HitTripletGeneratorFromPairAndLayersFactory;
 
#endif
