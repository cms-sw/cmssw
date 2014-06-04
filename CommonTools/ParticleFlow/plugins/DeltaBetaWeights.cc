#include "CommonTools/ParticleFlow/plugins/DeltaBetaWeights.h"

DeltaBetaWeights::DeltaBetaWeights(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  chargedSrc_(iConfig.getParameter<edm::InputTag>("chargedFromPV")),
  puSrc_(iConfig.getParameter<edm::InputTag>("chargedFromPU"))
{
  
  produces<reco::PFCandidateCollection>();
  
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

  edm::Handle<reco::PFCandidateCollection> pfCharged;
  edm::Handle<reco::PFCandidateCollection> pfPU;
  edm::Handle<reco::PFCandidateCollection> src;

  iEvent.getByLabel(src_,src);
  iEvent.getByLabel(chargedSrc_,pfCharged);
  iEvent.getByLabel(puSrc_,pfPU);

  double sumNPU;
  double sumPU;

  std::auto_ptr<reco::PFCandidateCollection> out(new reco::PFCandidateCollection); 


  for (unsigned int i=0;i<src->size();++i) {
    if ((*src)[i].charge() !=0) {
      out->push_back((*src)[i]);
      continue;
    }

    sumNPU=0.0;
    sumPU=0.0;
    for (unsigned int j=0;j<pfCharged->size();++j) {
      sumNPU += 1./(deltaR2((*src)[i].eta(),
			    (*src)[i].phi(),
			    (*pfCharged)[j].eta(),
			    (*pfCharged)[j].phi()));
    }
    for (unsigned int j=0;j<pfPU->size();++j) {
      sumPU += 1./(deltaR2((*src)[i].eta(),
			   (*src)[i].phi(),
			   (*pfPU)[j].eta(),
			   (*pfPU)[j].phi()));
    }

    reco::PFCandidate neutral = (*src)[i];
    if (sumNPU+sumPU>0)
      neutral.setP4(((sumNPU)/(sumNPU+sumPU))*neutral.p4());
    out->push_back(neutral);
     
  }

  iEvent.put(out);
 
}
