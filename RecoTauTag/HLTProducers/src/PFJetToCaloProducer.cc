#include "RecoTauTag/HLTProducers/interface/PFJetToCaloProducer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//


PFJetToCaloProducer::PFJetToCaloProducer(const edm::ParameterSet& iConfig)
{

  tauSrc_ = iConfig.getParameter<edm::InputTag>("Source");
  produces<reco::CaloJetCollection>();
}

PFJetToCaloProducer::~PFJetToCaloProducer(){ }

void PFJetToCaloProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;

  CaloJet::Specific specific;
  
  CaloJetCollection * jetCollectionTmp = new CaloJetCollection;
  edm::Handle<PFJetCollection> tauJets;
    iEvent.getByLabel( tauSrc_, tauJets );
    PFJetCollection::const_iterator i = tauJets->begin();
    for(;i !=tauJets->end(); i++ ) {

      //      cout <<"Jet Tracks"<<i->chargedHadronMultiplicity()<<std::endl;
      CaloJet jet(i->p4(),i->vertex(),specific);
      jet.setPdgId(15);
      jetCollectionTmp->push_back(jet);
     }


  
  auto_ptr<reco::CaloJetCollection> selectedTaus(jetCollectionTmp);

  iEvent.put(selectedTaus);


}
