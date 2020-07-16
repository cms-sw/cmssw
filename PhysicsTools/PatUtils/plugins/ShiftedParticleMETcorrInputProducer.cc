#include "PhysicsTools/PatUtils/plugins/ShiftedParticleMETcorrInputProducer.h"

ShiftedParticleMETcorrInputProducer::ShiftedParticleMETcorrInputProducer(const edm::ParameterSet& cfg)
    : srcOriginalToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("srcOriginal"))),
      srcShiftedToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("srcShifted"))) {
  edm::InputTag srcWeights = cfg.getParameter<edm::InputTag>("srcWeights");
  if (!srcWeights.label().empty())
    weightsToken_ = consumes<edm::ValueMap<float>>(srcWeights);

  produces<CorrMETData>();
}

ShiftedParticleMETcorrInputProducer::~ShiftedParticleMETcorrInputProducer() {
  // nothing to be done yet...
}

void ShiftedParticleMETcorrInputProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  edm::Handle<CandidateView> originalParticles;
  evt.getByToken(srcOriginalToken_, originalParticles);

  edm::Handle<CandidateView> shiftedParticles;
  evt.getByToken(srcShiftedToken_, shiftedParticles);

  edm::Handle<edm::ValueMap<float>> weights;
  if (!weightsToken_.isUninitialized())
    evt.getByToken(weightsToken_, weights);

  auto metCorrection = std::make_unique<CorrMETData>();
  if ((!weightsToken_.isUninitialized()) && (originalParticles->size() != shiftedParticles->size()))
    throw cms::Exception("InvalidInput")
        << "Original collection and shifted collection are of different size in ShiftedParticleMETcorrInputProducer\n";
  for (unsigned i = 0; i < originalParticles->size(); ++i) {
    float weight = 1.0;
    if (!weightsToken_.isUninitialized()) {
      edm::Ptr<reco::Candidate> particlePtr = originalParticles->ptrAt(i);
      while (!weights->contains(particlePtr.id()) && (particlePtr->numberOfSourceCandidatePtrs() > 0))
        particlePtr = particlePtr->sourceCandidatePtr(0);
      weight = (*weights)[particlePtr];
    }
    const reco::Candidate& originalParticle = originalParticles->at(i);
    metCorrection->mex += originalParticle.px() * weight;
    metCorrection->mey += originalParticle.py() * weight;
    metCorrection->sumet -= originalParticle.et() * weight;
  }
  for (unsigned i = 0; i < shiftedParticles->size(); ++i) {
    float weight = 1.0;
    if (!weightsToken_.isUninitialized()) {
      edm::Ptr<reco::Candidate> particlePtr = originalParticles->ptrAt(i);
      while (!weights->contains(particlePtr.id()) && (particlePtr->numberOfSourceCandidatePtrs() > 0))
        particlePtr = particlePtr->sourceCandidatePtr(0);
      weight = (*weights)[particlePtr];
    }
    const reco::Candidate& shiftedParticle = shiftedParticles->at(i);
    metCorrection->mex -= shiftedParticle.px() * weight;
    metCorrection->mey -= shiftedParticle.py() * weight;
    metCorrection->sumet += shiftedParticle.et() * weight;
  }

  evt.put(std::move(metCorrection));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedParticleMETcorrInputProducer);
