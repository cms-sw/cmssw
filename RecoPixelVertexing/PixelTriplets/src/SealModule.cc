#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletHLTGenerator.h"
DEFINE_SEAL_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, PixelTripletHLTGenerator, "PixelTripletHLTGenerator");

