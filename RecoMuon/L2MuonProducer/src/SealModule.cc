#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L2MuonProducer/src/L2MuonProducer.h"
#include "RecoMuon/L2MuonProducer/src/L2MuonCandidateProducer.h"


DEFINE_FWK_MODULE(L2MuonProducer);
DEFINE_FWK_MODULE(L2MuonCandidateProducer);
