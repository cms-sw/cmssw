#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
typedef BXVectorSimpleFlatTableProducer<l1t::CaloTower> SimpleTriggerL1CaloTowerFlatTableProducer;

#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
typedef BXVectorSimpleFlatTableProducer<l1t::CaloCluster> SimpleTriggerL1CaloClusterFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleTriggerL1CaloTowerFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1CaloClusterFlatTableProducer);
