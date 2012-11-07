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
#include "RecoEcal/EgammaClusterProducers/interface/ReducedRecHitCollectionProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/ReducedESRecHitCollectionProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/InterestingDetIdCollectionProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/Multi5x5ClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/Multi5x5SuperClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/CosmicClusterProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/EcalDigiSelector.h"
#include "RecoEcal/EgammaClusterProducers/interface/UncleanSCRecoveryProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/UnifiedSCCollectionProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/CleanAndMergeProducer.h"
#include "RecoEcal/EgammaClusterProducers/interface/PFSuperClusterProducer.h"

DEFINE_FWK_MODULE(IslandClusterProducer);
DEFINE_FWK_MODULE(HybridClusterProducer);
DEFINE_FWK_MODULE(SuperClusterProducer);
DEFINE_FWK_MODULE(EgammaSCCorrectionMaker);
DEFINE_FWK_MODULE(EgammaSimpleAnalyzer);
DEFINE_FWK_MODULE(PreshowerClusterProducer);
DEFINE_FWK_MODULE(PreshowerAnalyzer);
DEFINE_FWK_MODULE(RecHitFilter);
DEFINE_FWK_MODULE(PreshowerClusterShapeProducer);
DEFINE_FWK_MODULE(Multi5x5ClusterProducer);
DEFINE_FWK_MODULE(Multi5x5SuperClusterProducer);
DEFINE_FWK_MODULE(ReducedRecHitCollectionProducer);
DEFINE_FWK_MODULE(ReducedESRecHitCollectionProducer);
DEFINE_FWK_MODULE(InterestingDetIdCollectionProducer);
DEFINE_FWK_MODULE(CosmicClusterProducer);
DEFINE_FWK_MODULE(EcalDigiSelector);
DEFINE_FWK_MODULE(UncleanSCRecoveryProducer);
DEFINE_FWK_MODULE(UnifiedSCCollectionProducer);
DEFINE_FWK_MODULE(CleanAndMergeProducer);
DEFINE_FWK_MODULE(PFSuperClusterProducer);
