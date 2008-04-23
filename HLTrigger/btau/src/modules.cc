#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "L3MumuTrackingRegion.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, L3MumuTrackingRegion, "L3MumuTrackingRegion");

#include "HLTrigger/btau/interface/HLTJetTag.h"
DEFINE_FWK_MODULE(HLTJetTag);

#include "HLTrigger/btau/interface/HLTDisplacedmumuFilter.h"
DEFINE_FWK_MODULE(HLTDisplacedmumuFilter);

#include "HLTrigger/btau/interface/HLTmmkFilter.h"
DEFINE_FWK_MODULE(HLTmmkFilter);

#include "HLTrigger/btau/interface/GetJetsFromHLTobject.h"
DEFINE_FWK_MODULE(GetJetsFromHLTobject);
