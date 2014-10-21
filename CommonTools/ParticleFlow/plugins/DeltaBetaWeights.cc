#include "CommonTools/ParticleFlow/plugins/DeltaBetaWeights.h"

DeltaBetaWeights::DeltaBetaWeights(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  pfCharged_(iConfig.getParameter<edm::InputTag>("chargedFromPV")),
  pfPU_(iConfig.getParameter<edm::InputTag>("chargedFromPU"))
{
  produces<reco::PFCandidateCollection>();

  pfCharged_token = consumes<edm::View<reco::Candidate> >(pfCharged_);
  pfPU_token = consumes<edm::View<reco::Candidate> >(pfPU_);
  src_token = consumes<edm::View<reco::Candidate> >(src_);

  // pfCharged_token = consumes<reco::PFCandidateCollection>(pfCharged_);
  // pfPU_token = consumes<reco::PFCandidateCollection>(pfPU_);
  // src_token = consumes<reco::PFCandidateCollection>(src_);

}


DeltaBetaWeights::~DeltaBetaWeights()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeltaBetaWeights::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle<edm::View<reco::Candidate> > pfCharged;
  edm::Handle<edm::View<reco::Candidate> > pfPU;
  edm::Handle<edm::View<reco::Candidate> > src;
  

  iEvent.getByToken(src_token,src);
  iEvent.getByToken(pfCharged_token,pfCharged);
  iEvent.getByToken(pfPU_token,pfPU);

  double sumNPU = .0;
  double sumPU =  .0;

  std::auto_ptr<reco::PFCandidateCollection> out(new reco::PFCandidateCollection); 


  for (const reco::Candidate & cand : *src) {
    if (cand.charge() !=0) {
      // this part of code should be executed only if input collection is not entirely composed of neutral candidates, i.e. never by default
      edm::LogWarning("DeltaBetaWeights") << "Trying to reweight charged particle... saving it to output collection without any change";
      out->emplace_back(cand.charge(),cand.p4(),reco::PFCandidate::ParticleType::X);
      (out->back()).setParticleType((out->back()).translatePdgIdToType(cand.pdgId()));
      continue;
    }

    sumNPU=1.0;
    sumPU=1.0;
    double eta=cand.eta();
    double phi=cand.phi();
    for (const reco::Candidate &chCand : *pfCharged ) {
      double sum = (chCand.pt()*chCand.pt())/(deltaR2(eta,phi,chCand.eta(),chCand.phi()));
      if(sum > 1.0) sumNPU *= sum;
    }
    sumNPU=0.5*log(sumNPU);

    for (const reco::Candidate &puCand : *pfPU ) {
      double sum = (puCand.pt()*puCand.pt())/(deltaR2(eta,phi,puCand.eta(),puCand.phi()));
      if(sum > 1.0) sumPU *= sum;
    }
    sumPU=0.5*log(sumPU);

    reco::PFCandidate neutral = reco::PFCandidate(cand.charge(),cand.p4(),reco::PFCandidate::ParticleType::X);
    neutral.setParticleType(neutral.translatePdgIdToType(cand.pdgId()));
    if (sumNPU+sumPU>0)
      neutral.setP4(((sumNPU)/(sumNPU+sumPU))*neutral.p4());
    out->push_back(neutral);

  }

  iEvent.put(out);
 
}
