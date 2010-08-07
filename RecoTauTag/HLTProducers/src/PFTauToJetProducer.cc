#include "RecoTauTag/HLTProducers/interface/PFTauToJetProducer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//


PFTauToJetProducer::PFTauToJetProducer(const edm::ParameterSet& iConfig)
{

  tauSrc_ = iConfig.getParameter<edm::InputTag>("Source");
  produces<reco::PFJetCollection>();
}

PFTauToJetProducer::~PFTauToJetProducer(){ }

void PFTauToJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  
  PFJetCollection * jetCollectionTmp = new PFJetCollection;
  edm::Handle<PFTauCollection> tauJets;
    iEvent.getByLabel( tauSrc_, tauJets );
    PFTauCollection::const_iterator i = tauJets->begin();
    for(;i !=tauJets->end(); i++ ) {
      PFJetRef pippo = i->pfTauTagInfoRef()->pfjetRef();
      if(pippo.isNonnull())
	jetCollectionTmp->push_back(*pippo);
      
    }


  
  std::auto_ptr<reco::PFJetCollection> selectedTaus(jetCollectionTmp);

  iEvent.put(selectedTaus);


}
