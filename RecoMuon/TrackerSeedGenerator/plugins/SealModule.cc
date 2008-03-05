#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "TSGFromOrderedHits.h"
#include "TSGSmart.h"
#include "TSGForRoadSearch.h"
#include "TSGFromPropagation.h"
#include "DualByHitFractionTSG.h"
#include "DualByEtaTSG.h"
#include "DualByZTSG.h"
#include "CombinedTSG.h"


DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGSmart, "TSGSmart");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromPropagation, "TSGFromPropagation");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, DualByHitFractionTSG, "DualByHitFractionTSG");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, DualByEtaTSG, "DualByEtaTSG");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, DualByZTSG, "DualByZTSG");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, CombinedTSG, "CombinedTSG");


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
