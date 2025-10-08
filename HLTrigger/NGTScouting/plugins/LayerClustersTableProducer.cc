#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
typedef SimpleCollectionFlatTableProducer<reco::CaloCluster> LayerClustersCollectionTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LayerClustersCollectionTableProducer);
