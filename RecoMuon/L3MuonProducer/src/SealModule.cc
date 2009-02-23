#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonProducer.h"
#include "RecoMuon/L3MuonProducer/src/L3TkMuonProducer.h"
#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L3MuonProducer);
DEFINE_ANOTHER_FWK_MODULE(L3TkMuonProducer);
DEFINE_ANOTHER_FWK_MODULE(L3MuonCandidateProducer);
