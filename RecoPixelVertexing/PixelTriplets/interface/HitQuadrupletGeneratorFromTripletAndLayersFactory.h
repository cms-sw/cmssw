#ifndef PixelTriplets_HitQuadrupletGeneratorFromTripletAndLayersFactory_H
#define PixelTriplets_HitQuadrupletGeneratorFromTripletAndLayersFactory_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet; class ConsumesCollector;}

typedef edmplugin::PluginFactory<HitQuadrupletGeneratorFromTripletAndLayers *(const edm::ParameterSet &, edm::ConsumesCollector&)>
	HitQuadrupletGeneratorFromTripletAndLayersFactory;

#endif
