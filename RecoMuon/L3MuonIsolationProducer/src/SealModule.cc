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

#include "RecoMuon/L3MuonIsolationProducer/src/MuonHLTHcalPFClusterIsolationProducer.h"
DEFINE_FWK_MODULE(MuonHLTHcalPFClusterIsolationProducer);

#include "RecoMuon/L3MuonIsolationProducer/src/MuonHLTEcalPFClusterIsolationProducer.h"
DEFINE_FWK_MODULE(MuonHLTEcalPFClusterIsolationProducer);

#include "RecoMuon/L3MuonIsolationProducer/src/L3MuonSumCaloPFIsolationProducer.h"
DEFINE_FWK_MODULE(L3MuonSumCaloPFIsolationProducer);
