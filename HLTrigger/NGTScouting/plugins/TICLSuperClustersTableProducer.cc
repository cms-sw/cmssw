#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
typedef SimpleCollectionFlatTableProducer<reco::SuperCluster> TICLSuperClustersTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLSuperClustersTableProducer);
