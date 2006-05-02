#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEcal/EgammaClusterProducers/interface/BumpProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/IslandClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/SuperClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(BumpProducer);
DEFINE_ANOTHER_FWK_MODULE(IslandClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(HybridClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(SuperClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(EgammaSCCorrectionMaker);
