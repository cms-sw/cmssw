#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"




#include "L3MuonIsolationProducer.h"
DEFINE_FWK_MODULE(L3MuonIsolationProducer);

#include "L3MuonCombinedRelativeIsolationProducer.h"
DEFINE_FWK_MODULE(L3MuonCombinedRelativeIsolationProducer);


#include "RecoMuon/L3MuonIsolationProducer/src/L3MuonSumCaloPFIsolationProducer.h"
DEFINE_FWK_MODULE(L3MuonSumCaloPFIsolationProducer);
