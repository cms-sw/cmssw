#include "RecoTauTag/HLTProducers/interface/PFTauToJetProducer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//

PFTauToJetProducer::PFTauToJetProducer(const edm::ParameterSet& iConfig) {
  tauSrc_ = consumes<reco::PFTauCollection>(iConfig.getParameter<edm::InputTag>("Source"));
  produces<reco::CaloJetCollection>();
}

PFTauToJetProducer::~PFTauToJetProducer() {}

void PFTauToJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iES) const {
  using namespace reco;
  using namespace edm;
  using namespace std;
  CaloJet::Specific specific;

  CaloJetCollection* jetCollectionTmp = new CaloJetCollection;
  edm::Handle<PFTauCollection> tauJets;
  iEvent.getByToken(tauSrc_, tauJets);
  PFTauCollection::const_iterator i = tauJets->begin();
  for (; i != tauJets->end(); i++) {
    CaloJet jet(i->p4(), i->vertex(), specific);
    jet.setPdgId(15);
    jetCollectionTmp->push_back(jet);
  }

  unique_ptr<reco::CaloJetCollection> selectedTaus(jetCollectionTmp);
  iEvent.put(std::move(selectedTaus));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTauToJetProducer);
