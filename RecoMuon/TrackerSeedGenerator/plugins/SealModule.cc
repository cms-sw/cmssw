#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorBC.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromOrderedHits.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGForRoadSearch.h"

DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TrackerSeedGeneratorBC, "TrackerSeedGeneratorBC");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");

