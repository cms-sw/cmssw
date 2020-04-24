#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "TSGFromOrderedHits.h"
#include "TSGSmart.h"
#include "TSGForRoadSearch.h"
#include "TSGFromPropagation.h"
#include "DualByEtaTSG.h"
#include "DualByL2TSG.h"
#include "CombinedTSG.h"


DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGSmart, "TSGSmart");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromPropagation, "TSGFromPropagation");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, DualByEtaTSG, "DualByEtaTSG");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, DualByL2TSG, "DualByL2TSG");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, CombinedTSG, "CombinedTSG");




#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "TSGFromL1Muon.h"
#include "TSGFromL2Muon.h"


DEFINE_FWK_MODULE(TSGFromL1Muon);
DEFINE_FWK_MODULE(TSGFromL2Muon);

#include "CollectionCombiner.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


typedef CollectionCombiner<std::vector< Trajectory> > TrajectoryCombiner;
typedef CollectionCombiner<L3MuonTrajectorySeedCollection> L3MuonTrajectorySeedCombiner;
typedef CollectionCombiner<reco::TrackCollection> L3TrackCombiner;
typedef CollectionCombiner<TrackCandidateCollection> L3TrackCandCombiner;
typedef CollectionCombiner<reco::MuonTrackLinksCollection> L3TrackLinksCombiner;

DEFINE_FWK_MODULE(TrajectoryCombiner);
DEFINE_FWK_MODULE(L3MuonTrajectorySeedCombiner);
DEFINE_FWK_MODULE(L3TrackCombiner);
DEFINE_FWK_MODULE(L3TrackCandCombiner);
DEFINE_FWK_MODULE(L3TrackLinksCombiner);

