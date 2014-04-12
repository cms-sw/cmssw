#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonProducer.h"
#include "RecoMuon/L3MuonProducer/src/L3TkMuonProducer.h"
#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducer.h"
#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducerFromMuons.h"

DEFINE_FWK_MODULE(L3MuonProducer);
DEFINE_FWK_MODULE(L3TkMuonProducer);
DEFINE_FWK_MODULE(L3MuonCandidateProducer);
DEFINE_FWK_MODULE(L3MuonCandidateProducerFromMuons);
