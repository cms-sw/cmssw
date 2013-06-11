#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/GEMRecHit/src/GEMRecHitProducer.h"
// #include "RecoLocalMuon/GEMRecHit/interface/GEMPointProducer.h"
// #include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitAli.h"

#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitAlgoFactory.h"
#include "RecoLocalMuon/GEMRecHit/src/GEMRecHitStandardAlgo.h"


DEFINE_FWK_MODULE(GEMRecHitProducer);
// DEFINE_FWK_MODULE(GEMPointProducer);
// DEFINE_FWK_MODULE(GEMRecHitAli);
DEFINE_EDM_PLUGIN (GEMRecHitAlgoFactory, GEMRecHitStandardAlgo, "GEMRecHitStandardAlgo");
