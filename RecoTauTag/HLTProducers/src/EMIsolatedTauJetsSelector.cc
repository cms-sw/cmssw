#include "RecoTauTag/HLTProducers/interface/EMIsolatedTauJetsSelector.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//


EMIsolatedTauJetsSelector::EMIsolatedTauJetsSelector(const edm::ParameterSet& iConfig)
{
  singleTau      = iConfig.getParameter<edm::InputTag>("SingleTau");
  doubleTau       = iConfig.getParameter<edm::InputTag>("DoubleTau");
  l1Code     = iConfig.getParameter<edm::InputTag>("L1Code"); 
 
  produces<reco::CaloJetCollection>("Isolated");
  produces<reco::CaloJetCollection>("NotIsolated");
}

EMIsolatedTauJetsSelector::~EMIsolatedTauJetsSelector(){ }

void EMIsolatedTauJetsSelector::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  

  CaloJetCollection * jetIsolatedCollection = new CaloJetCollection;
  CaloJetCollection * jetNotIsolatedCollection = new CaloJetCollection;
 

  Handle< vector<int> > l1Decision;
  iEvent.getByLabel(l1Code,l1Decision);

  int l1Deco = -1000;
   l1Deco = *(l1Decision->begin());

  if(l1Deco == 1 || l1Deco == 2) {
    //SingleTau fired but NOT DoubleTau So SingleTau Jet should be isolated
    edm::Handle<EMIsolatedTauTagInfoCollection> singleTauJets;
    iEvent.getByLabel( singleTau, singleTauJets );
    EMIsolatedTauTagInfoCollection::const_iterator i = singleTauJets->begin();
    for(;i !=singleTauJets->end(); i++ ) {
      double discriminator = (*i).discriminator();
      if(discriminator > 0) {
	const CaloJet* pippo = dynamic_cast<const CaloJet*>(&(i->jet()));
	jetIsolatedCollection->push_back(*pippo );
      }
      const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(i->jet()));
      jetNotIsolatedCollection->push_back(*notPippo );
    }
    
    //All DoubleTau jets are put into the non isolated collection
    edm::Handle<EMIsolatedTauTagInfoCollection> doubleTauJets;
    iEvent.getByLabel( doubleTau, doubleTauJets );
    EMIsolatedTauTagInfoCollection::const_iterator ii = doubleTauJets->begin();
    for(;ii !=singleTauJets->end(); ii++ ) {

      const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(ii->jet()));
      jetNotIsolatedCollection->push_back(*notPippo );
    }
    
  }
  if(l1Deco == 3) {
    //SingleTau NOT fired
    edm::Handle<EMIsolatedTauTagInfoCollection> singleTauJets;
    iEvent.getByLabel( singleTau, singleTauJets );
    EMIsolatedTauTagInfoCollection::const_iterator i = singleTauJets->begin();
    for(;i !=singleTauJets->end(); i++ ) {
      double discriminator = (*i).discriminator();
      if(discriminator > 0) {
	const CaloJet* pippo = dynamic_cast<const CaloJet*>(&(i->jet()));
	jetIsolatedCollection->push_back(*pippo );
      }
      const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(i->jet()));
      jetNotIsolatedCollection->push_back(*notPippo );
    }
    
    edm::Handle<EMIsolatedTauTagInfoCollection> doubleTauJets;
    iEvent.getByLabel( doubleTau, doubleTauJets );
    EMIsolatedTauTagInfoCollection::const_iterator ii = doubleTauJets->begin();
    for(;ii !=singleTauJets->end(); ii++ ) {
      double discriminator = (*ii).discriminator();
      if(discriminator > 0) {
	const CaloJet* pippo = dynamic_cast<const CaloJet*>(&(i->jet()));
	jetIsolatedCollection->push_back(*pippo );
      }
      
      const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(ii->jet()));
      jetNotIsolatedCollection->push_back(*notPippo );
    }
    
  }

 auto_ptr<reco::CaloJetCollection> isolatedTaus(jetIsolatedCollection);
 auto_ptr<reco::CaloJetCollection> notIsolatedTaus(jetNotIsolatedCollection);

  iEvent.put(isolatedTaus, "Isolated");
  iEvent.put(notIsolatedTaus,"NotIsolated");

}
