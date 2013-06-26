#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"




#include "L3MuonIsolationProducer.h"
DEFINE_FWK_MODULE(L3MuonIsolationProducer);

#include "L3MuonCombinedRelativeIsolationProducer.h"
DEFINE_FWK_MODULE(L3MuonCombinedRelativeIsolationProducer);

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "IsolationRegionAroundL3Muon.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, IsolationRegionAroundL3Muon, "IsolationRegionAroundL3Muon");

