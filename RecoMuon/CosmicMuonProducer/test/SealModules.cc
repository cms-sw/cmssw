#include "FWCore/Framework/interface/MakerMacros.h"
 
//define this as a plug-in
#include "RecoMuon/CosmicMuonProducer/test/CosmicMuonValidator.cc"
#include "RecoMuon/CosmicMuonProducer/test/RealCosmicDataAnalyzer.cc"


DEFINE_FWK_MODULE(CosmicMuonValidator);
DEFINE_FWK_MODULE(RealCosmicDataAnalyzer);
