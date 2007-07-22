#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

// Producer
#include "PixelTrackProducerWithZPos.h"
DEFINE_ANOTHER_FWK_MODULE(PixelTrackProducerWithZPos);

// Remover
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SiPixelRecHitRemover.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelRecHitRemover);

// Region
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "GlobalTrackingRegionProducerWithVertices.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, GlobalTrackingRegionProducerWithVertices, "GlobalRegionProducerWithVertices");

// Generator
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTripletLowPtGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, PixelTripletLowPtGenerator,"PixelTripletLowPtGenerator");

// Filter
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilterFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
DEFINE_EDM_PLUGIN(TrackHitsFilterFactory, ClusterShapeTrackFilter, "ClusterShapeTrackFilter");

// Fitter
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelFitterByHelixProjections.h"
DEFINE_EDM_PLUGIN(PixelFitterFactory, LowPtPixelFitterByHelixProjections, "LowPtPixelFitterByHelixProjections");

// Cleaner
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelTrackCleanerBySharedHits.h"
DEFINE_EDM_PLUGIN(PixelTrackCleanerFactory, LowPtPixelTrackCleanerBySharedHits, "LowPtPixelTrackCleanerBySharedHits");

// Seed
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTrackSeedProducer.h"
DEFINE_ANOTHER_FWK_MODULE(PixelTrackSeedProducer);
