#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

// Cluster Vertexing
#include "HIPixelClusterVtxProducer.h"
DEFINE_FWK_MODULE(HIPixelClusterVtxProducer);

// Median Vertexing
#include "HIPixelMedianVtxProducer.h"
DEFINE_FWK_MODULE(HIPixelMedianVtxProducer);

// Best Vertex Producer
#include "RecoHI/HiTracking/interface/HIBestVertexProducer.h"
DEFINE_FWK_MODULE(HIBestVertexProducer);

// Restricted HI tracking regions               
#include "HITrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HITrackingRegionProducer, "HITrackingRegionProducer");

#include "HITrackingRegionForPrimaryVtxProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HITrackingRegionForPrimaryVtxProducer, "HITrackingRegionForPrimaryVtxProducer");

// Pixel track filter
#include "RecoHI/HiTracking/interface/HIPixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
DEFINE_EDM_PLUGIN(PixelTrackFilterWithESFactory, HIPixelTrackFilter, "HIPixelTrackFilter");

// Pixel prototrack filter
#include "RecoHI/HiTracking/interface/HIProtoTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
DEFINE_EDM_PLUGIN(PixelTrackFilterFactory, HIProtoTrackFilter, "HIProtoTrackFilter");
