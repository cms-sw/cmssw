#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "RecoParticleFlow/PFTracking/interface/VertexFilter.h"
#include "RecoParticleFlow/PFTracking/interface/LightPFTrackProducer.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GoodSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(PFElecTkProducer);
DEFINE_ANOTHER_FWK_MODULE(VertexFilter);
DEFINE_ANOTHER_FWK_MODULE(LightPFTrackProducer);
