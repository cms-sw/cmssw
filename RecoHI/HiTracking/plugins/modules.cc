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
