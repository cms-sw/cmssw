#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/btau/interface/HLTJetTag.h"
#include "HLTrigger/btau/interface/HLTTauL25DoubleFilter.h"
#include "HLTrigger/btau/interface/HLTDisplacedmumuFilter.h"
#include "HLTrigger/btau/interface/HLTmumuGammaFilter.h"
#include "HLTrigger/btau/interface/GetJetsFromHLTobject.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "L3MumuTrackingRegion.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, L3MumuTrackingRegion, "L3MumuTrackingRegion");
DEFINE_FWK_MODULE(HLTJetTag);
DEFINE_FWK_MODULE(HLTTauL25DoubleFilter);
DEFINE_FWK_MODULE(HLTDisplacedmumuFilter);
DEFINE_FWK_MODULE(HLTmumuGammaFilter);
DEFINE_FWK_MODULE(GetJetsFromHLTobject);
