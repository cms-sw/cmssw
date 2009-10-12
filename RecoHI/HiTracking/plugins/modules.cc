#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

DEFINE_SEAL_MODULE();

// Median Vertexing
#include "HIPixelMedianVtxProducer.h"
DEFINE_ANOTHER_FWK_MODULE(HIPixelMedianVtxProducer);

// Restricted HI tracking regions                                                                                                              
#include "HITrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HITrackingRegionProducer, "HITrackingRegionProducer");

#include "HITrackingRegionForPrimaryVtxProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HITrackingRegionForPrimaryVtxProducer, "HITrackingRegionForPrimaryVtxProducer");
