#include "PhysicsTools/PatUtils/plugins/ShiftedParticleMETcorrInputProducer.h"

ShiftedParticleMETcorrInputProducer::ShiftedParticleMETcorrInputProducer(const edm::ParameterSet& cfg)
  : srcOriginalToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("srcOriginal")))
  , srcShiftedToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("srcShifted")))
{
  produces<CorrMETData>();
}

ShiftedParticleMETcorrInputProducer::~ShiftedParticleMETcorrInputProducer()
{
// nothing to be done yet...
}

void ShiftedParticleMETcorrInputProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const
{
  edm::Handle<CandidateView> originalParticles;
  evt.getByToken(srcOriginalToken_, originalParticles);

  edm::Handle<CandidateView> shiftedParticles;
  evt.getByToken(srcShiftedToken_, shiftedParticles);

  std::auto_ptr<CorrMETData> metCorrection(new CorrMETData());

  for ( CandidateView::const_iterator originalParticle = originalParticles->begin();
	originalParticle != originalParticles->end(); ++originalParticle ) {
    metCorrection->mex   += originalParticle->px();
    metCorrection->mey   += originalParticle->py();
    metCorrection->sumet += originalParticle->et();
  }

  for ( CandidateView::const_iterator shiftedParticle = shiftedParticles->begin();
	shiftedParticle != shiftedParticles->end(); ++shiftedParticle ) {
    metCorrection->mex   -= shiftedParticle->px();
    metCorrection->mey   -= shiftedParticle->py();
    metCorrection->sumet -= shiftedParticle->et();
  }

  evt.put(metCorrection);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedParticleMETcorrInputProducer);


