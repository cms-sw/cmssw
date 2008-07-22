#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleIsoDetIdCollectionProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/GamIsoDetIdCollectionProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(GamIsoDetIdCollectionProducer);
DEFINE_ANOTHER_FWK_MODULE(EleIsoDetIdCollectionProducer);


