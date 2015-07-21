#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

HitTripletGeneratorFromPairAndLayers::HitTripletGeneratorFromPairAndLayers(unsigned int maxElement):
  theLayerCache(nullptr),
  theMaxElement(maxElement)
{}

HitTripletGeneratorFromPairAndLayers::HitTripletGeneratorFromPairAndLayers(const edm::ParameterSet& pset):
  HitTripletGeneratorFromPairAndLayers(pset.getParameter<unsigned int>("maxElement"))
{}

HitTripletGeneratorFromPairAndLayers::~HitTripletGeneratorFromPairAndLayers() {}

void HitTripletGeneratorFromPairAndLayers::init(std::unique_ptr<HitPairGeneratorFromLayerPair>&& pairGenerator, LayerCacheType *layerCache) {
  thePairGenerator = std::move(pairGenerator);
  theLayerCache = layerCache;
}
