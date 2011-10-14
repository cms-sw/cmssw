#include "PhysicsTools/PatUtils/plugins/ShiftedParticleMETcorrInputProducer.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

ShiftedParticleMETcorrInputProducer::ShiftedParticleMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  srcOriginal_ = cfg.getParameter<edm::InputTag>("srcOriginal");
  srcShifted_ = cfg.getParameter<edm::InputTag>("srcShifted");
  
  produces<CorrMETData>();
}
 
ShiftedParticleMETcorrInputProducer::~ShiftedParticleMETcorrInputProducer()
{
// nothing to be done yet...
}
    
void ShiftedParticleMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  typedef edm::View<reco::Candidate> CandidateView;

  edm::Handle<CandidateView> originalParticles;
  evt.getByLabel(srcOriginal_, originalParticles);

  edm::Handle<CandidateView> shiftedParticles;
  evt.getByLabel(srcShifted_, shiftedParticles);

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
 

