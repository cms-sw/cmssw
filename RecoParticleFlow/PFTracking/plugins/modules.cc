#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "RecoParticleFlow/PFTracking/interface/LightPFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFNuclearProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFConversionsProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFV0Producer.h"
#include "RecoParticleFlow/PFTracking/interface/ConvBremSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/ElectronSeedMerger.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GoodSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(PFElecTkProducer);
DEFINE_ANOTHER_FWK_MODULE(LightPFTrackProducer);
DEFINE_ANOTHER_FWK_MODULE(PFNuclearProducer);
DEFINE_ANOTHER_FWK_MODULE(PFConversionsProducer);
DEFINE_ANOTHER_FWK_MODULE(PFV0Producer);
DEFINE_ANOTHER_FWK_MODULE(ConvBremSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(ElectronSeedMerger);
