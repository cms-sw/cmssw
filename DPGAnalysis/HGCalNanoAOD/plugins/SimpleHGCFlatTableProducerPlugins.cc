#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
typedef SimpleFlatTableProducer<SimCluster> SimpleSimClusterFlatTableProducer;

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
typedef SimpleFlatTableProducer<PCaloHit> SimplePCaloHitFlatTableProducer;
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
typedef SimpleFlatTableProducer<CaloRecHit> SimpleCaloRecHitFlatTableProducer;

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
typedef SimpleFlatTableProducer<CaloParticle> SimpleCaloParticleFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimplePCaloHitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCaloRecHitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSimClusterFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCaloParticleFlatTableProducer);
