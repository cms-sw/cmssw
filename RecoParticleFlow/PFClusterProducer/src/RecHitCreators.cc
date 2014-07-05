#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFEcalRecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalRecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHFRecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFPSRecHitCreator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHGCalRecHitCreator.h"


EDM_REGISTER_PLUGINFACTORY(PFRecHitFactory, "PFRecHitFactory");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFEERecHitCreator, "PFEERecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFEKRecHitCreator, "PFEKRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFEBRecHitCreator, "PFEBRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHERecHitCreator, "PFHERecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHBRecHitCreator, "PFHBRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHBHERecHitCreator, "PFHBHERecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHORecHitCreator, "PFHORecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHFRecHitCreator, "PFHFRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFPSRecHitCreator, "PFPSRecHitCreator");

DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHGCEERecHitCreator, "PFHGCEERecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHGCHEFRecHitCreator, "PFHGCHEFRecHitCreator");
DEFINE_EDM_PLUGIN(PFRecHitFactory, PFHGCHEBRecHitCreator, "PFHGCHEBRecHitCreator");
