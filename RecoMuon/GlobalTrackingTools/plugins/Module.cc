#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/GlobalTrackingTools/plugins/GlobalTrackQualityProducer.h"


DEFINE_FWK_MODULE(GlobalTrackQualityProducer);

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, MuonTrackingRegionBuilder, "MuonTrackingRegionBuilder");

