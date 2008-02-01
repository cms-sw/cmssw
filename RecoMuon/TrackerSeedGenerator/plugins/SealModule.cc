#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromOrderedHits.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGSmart.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGForRoadSearch.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromPropagation.h"

DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGSmart, "TSGSmart");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromPropagation, "TSGFromPropagation");

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromL1Muon.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TSGFromL1Muon);
