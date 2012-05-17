#include "FWCore/Framework/interface/MakerMacros.h"
#include "FastSimulation/Muons/plugins/FastTSGFromL2Muon.h"
#include "FastSimulation/Muons/plugins/FastTSGFromPropagation.h"
#include "FastSimulation/Muons/plugins/FastTSGFromIOHit.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"


DEFINE_FWK_MODULE(FastTSGFromL2Muon);
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, FastTSGFromPropagation, "FastTSGFromPropagation");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, FastTSGFromIOHit, "FastTSGFromIOHit");

