//#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "HIMuonTrackingRegionProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HIMuonTrackingRegionProducer, "HIMuonTrackingRegionProducer");
