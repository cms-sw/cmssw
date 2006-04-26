#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEcal/EgammaClusterProducers/interface/BumpProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/TestClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/TestSCProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(BumpProducer);
DEFINE_ANOTHER_FWK_MODULE(TestClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(TestSCProducer);
DEFINE_ANOTHER_FWK_MODULE(HybridClusterProducer);
