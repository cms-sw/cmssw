#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromOrderedHits.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGForRoadSearch.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromPropagation.h"

DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromPropagation, "TSGFromPropagation");
