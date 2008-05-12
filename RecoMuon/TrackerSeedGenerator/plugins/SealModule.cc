#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "TSGFromOrderedHits.h"
#include "TSGSmart.h"
#include "TSGForRoadSearch.h"
#include "TSGFromPropagation.h"
#include "DualByEtaTSG.h"
#include "CombinedTSG.h"


DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGSmart, "TSGSmart");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromPropagation, "TSGFromPropagation");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, DualByEtaTSG, "DualByEtaTSG");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, CombinedTSG, "CombinedTSG");

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonPixelTrackFitter.h"
DEFINE_EDM_PLUGIN(PixelFitterFactory, L1MuonPixelTrackFitter, "L1MuonPixelTrackFitter");

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonRegionProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, L1MuonRegionProducer, "L1MuonRegionProducer");



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "TSGFromL1Muon.h"
#include "TSGFromL2Muon.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TSGFromL1Muon);
DEFINE_ANOTHER_FWK_MODULE(TSGFromL2Muon);

#include "CollectionCombiner.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
typedef CollectionCombiner<std::vector< Trajectory> > TrajectoryCombiner;
typedef CollectionCombiner<L3MuonTrajectorySeedCollection> L3MuonTrajectorySeedCombiner;

DEFINE_ANOTHER_FWK_MODULE(TrajectoryCombiner);
DEFINE_ANOTHER_FWK_MODULE(L3MuonTrajectorySeedCombiner);
