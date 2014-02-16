#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterProducer.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFHCALSuperClusterProducer.h"


DEFINE_FWK_MODULE(PFClusterProducer);
DEFINE_FWK_MODULE(PFHCALSuperClusterProducer);


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitDualNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCaloNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCaloTowerNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitHCALNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFECALHashNavigator.h"


EDM_REGISTER_PLUGINFACTORY(PFRecHitNavigationFactory, "PFRecHitNavigationFactory");

DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalBarrelNavigator, "PFRecHitEcalBarrelNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalEndcapNavigator, "PFRecHitEcalEndcapNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFECALHashNavigator, "PFECALHashNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitEcalNavigator, "PFRecHitEcalNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitCaloTowerNavigator, "PFRecHitCaloTowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitPreshowerNavigator, "PFRecHitPreshowerNavigator");
DEFINE_EDM_PLUGIN(PFRecHitNavigationFactory, PFRecHitHCALNavigator, "PFRecHitHCALNavigator");


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitQTestBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitQTests.h"
EDM_REGISTER_PLUGINFACTORY(PFRecHitQTestFactory, "PFRecHitQTestFactory");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestThreshold, "PFRecHitQTestThreshold");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHOThreshold, "PFRecHitQTestHOThreshold");

DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestECAL, "PFRecHitQTestECAL");
DEFINE_EDM_PLUGIN(PFRecHitQTestFactory, PFRecHitQTestHCALCalib29, "PFRecHitQTestHCALCalib29");


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFEcalRecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalRecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFPSRecHitCreator.h"


EDM_REGISTER_PLUGINFACTORY(PFRecHitFactory, "PFRecHitFactory");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFEERecHitCreator, "PFEERecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFEBRecHitCreator, "PFEBRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHERecHitCreator, "PFHERecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHBRecHitCreator, "PFHBRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHORecHitCreator, "PFHORecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHFEMRecHitCreator, "PFHFEMRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHFHADRecHitCreator, "PFHFHADRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFPSRecHitCreator, "PFPSRecHitCreator");





#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducer.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFCTRecHitProducer.h"

DEFINE_FWK_MODULE(PFRecHitProducer);
DEFINE_FWK_MODULE(PFCTRecHitProducer);
