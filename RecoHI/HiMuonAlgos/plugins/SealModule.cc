//#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

using namespace cms;

#include "HIMuonTrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HIMuonTrackingRegionProducer,
                  "HIMuonTrackingRegionProducer");

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionEDProducerT.h"
using HIMuonTrackingRegionEDProducer =
    TrackingRegionEDProducerT<HIMuonTrackingRegionProducer>;
DEFINE_FWK_MODULE(HIMuonTrackingRegionEDProducer);
