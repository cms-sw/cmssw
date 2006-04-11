#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEcal/EgammaClusterProducers/interface/BumpProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/TestClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/SCTestAnalyzer.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(BumpProducer);
DEFINE_ANOTHER_FWK_MODULE(TestClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(SCTestAnalyzer);
