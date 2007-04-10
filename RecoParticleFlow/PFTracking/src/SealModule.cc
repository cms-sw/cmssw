#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GoodSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(PFElecTkProducer);

