#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "RecoParticleFlow/PFTracking/interface/LightPFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFNuclearProducer.h"
#include "RecoParticleFlow/PFTracking/plugins/PFConversionProducer.h"
#include "RecoParticleFlow/PFTracking/plugins/PFV0Producer.h"
#include "RecoParticleFlow/PFTracking/interface/ElectronSeedMerger.h"
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexCandidateProducer.h"
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexProducer.h"
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedTrackerVertexProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexSelector.cc"
#include "RecoParticleFlow/PFTracking/plugins/SimVertexSelector.cc"

DEFINE_FWK_MODULE(GoodSeedProducer);
DEFINE_FWK_MODULE(PFElecTkProducer);
DEFINE_FWK_MODULE(LightPFTrackProducer);
DEFINE_FWK_MODULE(PFNuclearProducer);
DEFINE_FWK_MODULE(PFConversionProducer);
DEFINE_FWK_MODULE(PFV0Producer);
DEFINE_FWK_MODULE(ElectronSeedMerger);
DEFINE_FWK_MODULE(PFDisplacedVertexCandidateProducer);
DEFINE_FWK_MODULE(PFDisplacedVertexProducer);
DEFINE_FWK_MODULE(PFDisplacedTrackerVertexProducer);
DEFINE_FWK_MODULE(PFDisplacedVertexSelector);
DEFINE_FWK_MODULE(SimVertexSelector);
DEFINE_FWK_MODULE(PFTrackProducer);

