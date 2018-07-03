#include "RecoParticleFlow/PFProducer/plugins/PFConcretePFCandidateProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


PFConcretePFCandidateProducer::PFConcretePFCandidateProducer(const edm::ParameterSet& iConfig) {
  inputColl_ = iConfig.getParameter<edm::InputTag>("src");
  // register products
  produces<reco::PFCandidateCollection>();
}

PFConcretePFCandidateProducer::~PFConcretePFCandidateProducer() {}

void PFConcretePFCandidateProducer::produce(edm::Event& iEvent, 
		    const edm::EventSetup& iSetup) {
  edm::Handle<reco::PFCandidateCollection> inputColl;
  bool inputOk = iEvent.getByLabel(inputColl_,inputColl);

  if (!inputOk){
    // nothing ... I guess we prefer to send an exception in the next lines
   }

  auto outputColl = std::make_unique<reco::PFCandidateCollection>();
  outputColl->resize(inputColl->size());
  
  for (unsigned int iCopy=0;iCopy!=inputColl->size();++iCopy){
    const reco::PFCandidate & pf=(*inputColl)[iCopy];
    (*outputColl)[iCopy]=pf;
    //dereferenced internally the ref and hardcopy the value
    (*outputColl)[iCopy].setVertex(pf.vertex());
    //math::XYZPoint(pf.vx(),pf.vy(),pf.vz()));
  }
  
  iEvent.put(std::move(outputColl));
}
