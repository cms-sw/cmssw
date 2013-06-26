#include "PhysicsTools/PatUtils/interface/ShiftedParticleProducerT.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

typedef ShiftedParticleProducerT<reco::PFCandidate> ShiftedPFCandidateProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPFCandidateProducer);

