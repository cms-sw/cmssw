#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitAlgoFactory.h"
#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitAlgoFactory.h"

#include "RecoLocalMuon/GEMRecHit/src/GEMRecHitProducer.h"
DEFINE_FWK_MODULE(GEMRecHitProducer);

#include "RecoLocalMuon/GEMRecHit/src/GEMRecHitStandardAlgo.h"
DEFINE_EDM_PLUGIN (GEMRecHitAlgoFactory, GEMRecHitStandardAlgo, "GEMRecHitStandardAlgo");


#include "RecoLocalMuon/GEMRecHit/src/ME0RecHitProducer.h"
DEFINE_FWK_MODULE(ME0RecHitProducer);

#include "RecoLocalMuon/GEMRecHit/src/ME0RecHitStandardAlgo.h"
DEFINE_EDM_PLUGIN (ME0RecHitAlgoFactory, ME0RecHitStandardAlgo, "ME0RecHitStandardAlgo");

