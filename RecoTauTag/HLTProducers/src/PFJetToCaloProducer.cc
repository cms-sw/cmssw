#include "RecoTauTag/HLTProducers/interface/PFJetToCaloProducer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//

PFJetToCaloProducer::PFJetToCaloProducer(const edm::ParameterSet& iConfig) {
  tauSrc_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("Source"));
  produces<reco::CaloJetCollection>();
}

PFJetToCaloProducer::~PFJetToCaloProducer() {}

void PFJetToCaloProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iES) const {
  using namespace reco;
  using namespace edm;
  using namespace std;

  std::unique_ptr<reco::CaloJetCollection> selectedTaus(new CaloJetCollection);

  edm::Handle<PFJetCollection> tauJets;
  iEvent.getByToken(tauSrc_, tauJets);

  CaloJet::Specific specific;
  for (PFJetCollection::const_iterator i = tauJets->begin(); i != tauJets->end(); ++i) {
    CaloJet jet(i->p4(), i->vertex(), specific);
    jet.setPdgId(15);
    selectedTaus->push_back(jet);
  }

  iEvent.put(std::move(selectedTaus));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetToCaloProducer);
