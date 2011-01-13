
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "RecoParticleFlow/PFProducer/plugins/PFProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFElectronTranslator.h"
#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFBlockElementSuperClusterProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/EFilter.h"


DEFINE_FWK_MODULE(PFProducer);
DEFINE_FWK_MODULE(PFElectronTranslator);
DEFINE_FWK_MODULE(PFBlockProducer);
DEFINE_FWK_MODULE(PFBlockElementSuperClusterProducer);
DEFINE_FWK_MODULE(EFilter);
