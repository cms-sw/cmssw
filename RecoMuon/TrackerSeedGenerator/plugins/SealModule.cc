#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorBC.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromOrderedHits.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TSGForRoadSearch.h"

DEFINE_SEAL_PLUGIN(TrackerSeedGeneratorFactory, TrackerSeedGeneratorBC, "TrackerSeedGeneratorBC");
DEFINE_SEAL_PLUGIN(TrackerSeedGeneratorFactory, TSGFromOrderedHits, "TSGFromOrderedHits");
DEFINE_SEAL_PLUGIN(TrackerSeedGeneratorFactory, TSGForRoadSearch, "TSGForRoadSearch");

