#include "CommonTools/ParticleFlow/plugins/PFnoPileUp.h"

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace std;
using namespace edm;
using namespace reco;

PFnoPileUp::PFnoPileUp(const edm::ParameterSet& iConfig) {
  tokenCandidatesView_ = consumes<CandidateView>(iConfig.getParameter<InputTag>("candidates"));
  vertexAssociationQuality_ = iConfig.getParameter<int>("vertexAssociationQuality");
  tokenVertexAssociation_ = consumes<CandToVertex>(iConfig.getParameter<edm::InputTag>("vertexAssociation"));
  tokenVertexAssociationQuality_ =
      consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vertexAssociation"));
  produces<edm::PtrVector<reco::Candidate>>();
}

PFnoPileUp::~PFnoPileUp() {}

void PFnoPileUp::produce(Event& iEvent, const EventSetup& iSetup) {
  unique_ptr<edm::PtrVector<reco::Candidate>> pOutput(new edm::PtrVector<reco::Candidate>);
  Handle<CandidateView> candidateView;
  iEvent.getByToken(tokenCandidatesView_, candidateView);
  edm::Handle<edm::Association<reco::VertexCollection>> assoHandle;
  iEvent.getByToken(tokenVertexAssociation_, assoHandle);
  const edm::Association<reco::VertexCollection>* associatedPV = assoHandle.product();
  edm::Handle<edm::ValueMap<int>> assoQualityHandle;
  iEvent.getByToken(tokenVertexAssociationQuality_, assoQualityHandle);
  const edm::ValueMap<int>* associationQuality = assoQualityHandle.product();
  for (unsigned i = 0; i < candidateView->size(); i++) {
    const reco::VertexRef& PVOrig = (*associatedPV)[candidateView->ptrAt(i)];
    int quality = (*associationQuality)[candidateView->ptrAt(i)];
    if (!(PVOrig.isNonnull() && (PVOrig.key() > 0) && (quality >= vertexAssociationQuality_)))
      pOutput->push_back(candidateView->ptrAt(i));
  }
  iEvent.put(std::move(pOutput));
}

DEFINE_FWK_MODULE(PFnoPileUp);