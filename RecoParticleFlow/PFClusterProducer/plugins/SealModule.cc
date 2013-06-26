#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterProducer.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFHCALSuperClusterProducer.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerECAL.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerHCAL.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFHCALDualTimeRecHitProducer.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerHO.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerPS.h"




DEFINE_FWK_MODULE(PFClusterProducer);
DEFINE_FWK_MODULE(PFHCALSuperClusterProducer);
DEFINE_FWK_MODULE(PFRecHitProducerECAL);
DEFINE_FWK_MODULE(PFRecHitProducerHCAL);
DEFINE_FWK_MODULE(PFHCALDualTimeRecHitProducer);
DEFINE_FWK_MODULE(PFRecHitProducerHO);
DEFINE_FWK_MODULE(PFRecHitProducerPS);
