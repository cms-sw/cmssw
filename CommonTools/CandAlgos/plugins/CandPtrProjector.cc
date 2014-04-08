//
//

/**
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"


class CandPtrProjector : public edm::EDProducer{
  public:
    explicit CandPtrProjector(const edm::ParameterSet & iConfig);
    ~CandPtrProjector();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;
    virtual void endJob() override;

  private:
    edm::EDGetTokenT<edm::View<reco::Candidate> > candSrcToken_;
    edm::EDGetTokenT<edm::View<reco::Candidate> > vetoSrcToken_;
};

CandPtrProjector::CandPtrProjector(const edm::ParameterSet & iConfig):
  candSrcToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src"))),
  vetoSrcToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("veto")))
{
  produces<edm::PtrVector<reco::Candidate> >();
}

CandPtrProjector::~CandPtrProjector()
{
}

void
CandPtrProjector::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  using namespace edm;
  Handle<View<reco::Candidate> > cands;
  iEvent.getByToken(candSrcToken_, cands);
  Handle<View<reco::Candidate> > vetos;
  iEvent.getByToken(vetoSrcToken_, vetos);

  std::auto_ptr<PtrVector<reco::Candidate> > result(new PtrVector<reco::Candidate>());
  std::set<reco::CandidatePtr> vetoedPtrs;
  for(size_t i = 0; i< vetos->size();  ++i) {
   for(size_t j=0,n=(*vetos)[i].numberOfSourceCandidatePtrs(); j<n;j++ )    {
     vetoedPtrs.insert((*vetos)[i].sourceCandidatePtr(j));   
  }
  }
 for(size_t i = 0; i< cands->size();  ++i) {
    reco::CandidatePtr c =  cands->ptrAt(i);
    if(vetoedPtrs.find(c)==vetoedPtrs.end())
    {
      result->push_back(c);
    }
  }
  iEvent.put(result);
}

void CandPtrProjector::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandPtrProjector);
