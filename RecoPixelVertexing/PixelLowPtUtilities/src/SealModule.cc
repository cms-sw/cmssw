#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

// Generator
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTripletLowPtGenerator.h"
DEFINE_SEAL_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, PixelTripletLowPtGenerator,"PixelTripletLowPtGenerator");

// Fitter
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelFitterByHelixProjections.h"
DEFINE_SEAL_PLUGIN(PixelFitterFactory, LowPtPixelFitterByHelixProjections, "LowPtPixelFitterByHelixProjections");

// Cleaner
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelTrackCleanerBySharedHits.h"
DEFINE_SEAL_PLUGIN(PixelTrackCleanerFactory, LowPtPixelTrackCleanerBySharedHits, "LowPtPixelTrackCleanerBySharedHits");
