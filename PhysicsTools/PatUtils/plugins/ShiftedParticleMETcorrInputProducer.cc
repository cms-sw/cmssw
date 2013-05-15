#include "PhysicsTools/PatUtils/plugins/ShiftedParticleMETcorrInputProducer.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef edm::View<reco::Candidate> CandidateView;

ShiftedParticleMETcorrInputProducer::ShiftedParticleMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  srcOriginal_ = cfg.getParameter<edm::InputTag>("srcOriginal");
  srcShifted_ = cfg.getParameter<edm::InputTag>("srcShifted");
 
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
  
  produces<CorrMETData>();
}
 
ShiftedParticleMETcorrInputProducer::~ShiftedParticleMETcorrInputProducer()
{
// nothing to be done yet...
}
    
void ShiftedParticleMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<ShiftedParticleMETcorrInputProducer::produce>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  }

  edm::Handle<CandidateView> originalParticles;
  evt.getByLabel(srcOriginal_, originalParticles);

  edm::Handle<CandidateView> shiftedParticles;
  evt.getByLabel(srcShifted_, shiftedParticles);

  std::auto_ptr<CorrMETData> metCorrection(new CorrMETData());

  int idxOriginalParticle = 0;
  for ( CandidateView::const_iterator originalParticle = originalParticles->begin();
	originalParticle != originalParticles->end(); ++originalParticle ) {
    if ( verbosity_ ) {
      std::cout << "originalParticle #" << idxOriginalParticle << ": Pt = " << originalParticle->pt() << "," 
		<< " eta = " << originalParticle->eta() << ", phi = " << originalParticle->phi() 
		<< " (Px = " << originalParticle->px() << ", Py = " << originalParticle->py() << ")" << std::endl;
    }
    metCorrection->mex   += originalParticle->px();
    metCorrection->mey   += originalParticle->py();
    metCorrection->sumet += originalParticle->et();
    ++idxOriginalParticle;
  }

  int idxShiftedParticle = 0;
  for ( CandidateView::const_iterator shiftedParticle = shiftedParticles->begin();
	shiftedParticle != shiftedParticles->end(); ++shiftedParticle ) {
    if ( verbosity_ ) {
      std::cout << "shiftedParticle #" << idxShiftedParticle << ": Pt = " << shiftedParticle->pt() << "," 
		<< " eta = " << shiftedParticle->eta() << ", phi = " << shiftedParticle->phi() 
		<< " (Px = " << shiftedParticle->px() << ", Py = " << shiftedParticle->py() << ")" << std::endl;
    }
    metCorrection->mex   -= shiftedParticle->px();
    metCorrection->mey   -= shiftedParticle->py();
    metCorrection->sumet -= shiftedParticle->et();
    ++idxShiftedParticle;
  }

  evt.put(metCorrection);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedParticleMETcorrInputProducer);
 

