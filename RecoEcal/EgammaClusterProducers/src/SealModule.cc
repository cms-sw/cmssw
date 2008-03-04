#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEcal/EgammaClusterProducers/interface/IslandClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/SuperClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"
#include "RecoEcal/EgammaClusterProducers/interface/EgammaSimpleAnalyzer.h"
#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/PreshowerAnalyzer.h"
#include "RecoEcal/EgammaClusterProducers/interface/RecHitFilter.h"
#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterShapeProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(IslandClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(HybridClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(SuperClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(EgammaSCCorrectionMaker);
DEFINE_ANOTHER_FWK_MODULE(EgammaSimpleAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(PreshowerClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(PreshowerAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(RecHitFilter);
DEFINE_ANOTHER_FWK_MODULE(PreshowerClusterShapeProducer);
