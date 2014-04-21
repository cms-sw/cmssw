#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitQTestBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitQTests.h"

EDM_REGISTER_PLUGINFACTORY(PFRecHitQTestFactory, "PFRecHitQTestFactory");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestThreshold, "PFRecHitQTestThreshold");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHOThreshold, "PFRecHitQTestHOThreshold");

DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestECAL, "PFRecHitQTestECAL");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHCALCalib29, "PFRecHitQTestHCALCalib29");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHCALChannel, "PFRecHitQTestHCALChannel");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHCALTimeVsDepth, "PFRecHitQTestHCALTimeVsDepth");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHCALThresholdVsDepth, "PFRecHitQTestHCALThresholdVsDepth");

