#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

DEFINE_SEAL_MODULE();

// Producers
#include "PixelTrackProducerWithZPos.h"
DEFINE_ANOTHER_FWK_MODULE(PixelTrackProducerWithZPos);

#include "PixelVertexProducerMedian.h"
DEFINE_ANOTHER_FWK_MODULE(PixelVertexProducerMedian);

#include "PixelVertexProducerClusters.h"
DEFINE_ANOTHER_FWK_MODULE(PixelVertexProducerClusters);

#include "TrackListCombiner.h"
DEFINE_ANOTHER_FWK_MODULE(TrackListCombiner);

// Generator
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TripletGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, TripletGenerator,"TripletGenerator");

// Filters
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
DEFINE_EDM_PLUGIN(PixelTrackFilterWithESFactory, ClusterShapeTrackFilter, "ClusterShapeTrackFilter");

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ValidHitPairFilter.h"
DEFINE_EDM_PLUGIN(PixelTrackFilterWithESFactory, ValidHitPairFilter, "ValidHitPairFilter");

// Fitter
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackFitter.h"
DEFINE_EDM_PLUGIN(PixelFitterFactory, TrackFitter, "TrackFitter");

// Cleaner
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackCleaner.h"
DEFINE_EDM_PLUGIN(PixelTrackCleanerFactory, TrackCleaner, "TrackCleaner");

// Seed
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SeedProducer.h"
DEFINE_ANOTHER_FWK_MODULE(SeedProducer);

// TrajectoryFilter
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilterESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(ClusterShapeTrajectoryFilterESProducer);
