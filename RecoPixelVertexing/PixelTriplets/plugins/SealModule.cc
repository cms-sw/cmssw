#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "PixelTripletHLTGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, PixelTripletHLTGenerator, "PixelTripletHLTGenerator");

#include "PixelTripletLargeTipGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, PixelTripletLargeTipGenerator, "PixelTripletLargeTipGenerator");

#include "PixelTripletNoTipGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory,PixelTripletNoTipGenerator,"PixelTripletNoTipGenerator"); 

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "CombinedHitTripletGenerator.h"
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, CombinedHitTripletGenerator, "StandardHitTripletGenerator");

#include "CombinedHitQuadrupletGenerator.h"
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, CombinedHitQuadrupletGenerator, "CombinedHitQuadrupletGenerator");

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayersFactory.h"
#include "PixelQuadrupletGenerator.h"
DEFINE_EDM_PLUGIN(HitQuadrupletGeneratorFromTripletAndLayersFactory, PixelQuadrupletGenerator, "PixelQuadrupletGenerator");

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletEDProducerT.h"
using PixelTripletHLTEDProducer = HitTripletEDProducerT<PixelTripletHLTGenerator>;
DEFINE_FWK_MODULE(PixelTripletHLTEDProducer);

using PixelTripletLargeTipEDProducer = HitTripletEDProducerT<PixelTripletLargeTipGenerator>;
DEFINE_FWK_MODULE(PixelTripletLargeTipEDProducer);
