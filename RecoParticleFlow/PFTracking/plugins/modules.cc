#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "RecoParticleFlow/PFTracking/interface/LightPFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFNuclearProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFConversionsProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFV0Producer.h"
#include "RecoParticleFlow/PFTracking/interface/ElectronSeedMerger.h"
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexCandidateProducer.h"
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexProducer.h"


DEFINE_FWK_MODULE(GoodSeedProducer);
DEFINE_FWK_MODULE(PFElecTkProducer);
DEFINE_FWK_MODULE(LightPFTrackProducer);
DEFINE_FWK_MODULE(PFNuclearProducer);
DEFINE_FWK_MODULE(PFConversionsProducer);
DEFINE_FWK_MODULE(PFV0Producer);
DEFINE_FWK_MODULE(ElectronSeedMerger);
DEFINE_FWK_MODULE(PFDisplacedVertexCandidateProducer);
DEFINE_FWK_MODULE(PFDisplacedVertexProducer);
