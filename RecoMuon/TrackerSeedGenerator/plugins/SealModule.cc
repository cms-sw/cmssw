#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "TSGFromOrderedHits.h"
#include "TSGSmart.h"
#include "TSGForRoadSearch.h"
#include "TSGFromPropagation.h"

DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGSmart, "TSGSmart");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, TSGFromPropagation, "TSGFromPropagation");

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "TSGFromL1Muon.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TSGFromL1Muon);
