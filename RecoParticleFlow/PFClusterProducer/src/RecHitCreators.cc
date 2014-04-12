#include "FWCore/Framework/interface/MakerMacros.h"

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
