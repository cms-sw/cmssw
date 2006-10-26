#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

HitTripletGeneratorFromPairAndLayersFactory::HitTripletGeneratorFromPairAndLayersFactory() 
  : seal::PluginFactory<HitTripletGeneratorFromPairAndLayers * (const edm::ParameterSet & p)>("HitTripletGeneratorFromPairAndLayersFactory")
{ }

HitTripletGeneratorFromPairAndLayersFactory::~HitTripletGeneratorFromPairAndLayersFactory()
{ }

HitTripletGeneratorFromPairAndLayersFactory * HitTripletGeneratorFromPairAndLayersFactory::get() 
{
  static HitTripletGeneratorFromPairAndLayersFactory theHitTripletGeneratorFromPairAndLayersFactory;
  return & theHitTripletGeneratorFromPairAndLayersFactory;
}


