#include "FWCore/Framework/interface/MakerMacros.h"
#include "FastSimulation/Muons/plugins/FastL3MuonProducer.h"
#include "FastSimulation/Muons/plugins/FastTSGFromL2Muon.h"
#include "FastSimulation/Muons/plugins/FastL1MuonProducer.h"
#include "FastSimulation/Muons/plugins/FastTSGFromPropagation.h"
#include "FastSimulation/Muons/plugins/FastTSGFromIOHit.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"


DEFINE_FWK_MODULE(FastL1MuonProducer);
DEFINE_FWK_MODULE(FastTSGFromL2Muon);
DEFINE_FWK_MODULE(FastL3MuonProducer);
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, FastTSGFromPropagation, "FastTSGFromPropagation");
DEFINE_EDM_PLUGIN(TrackerSeedGeneratorFactory, FastTSGFromIOHit, "FastTSGFromIOHit");

