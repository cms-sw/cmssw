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
   //   cout <<"Crasha qui?"<<endl;
  if(l1Deco == 1 || l1Deco == 2) {
    //    cout <<"Crasha qui?1"<<endl;
    //SingleTau fired but NOT DoubleTau So SingleTau Jet should be isolated
    edm::Handle<EMIsolatedTauTagInfoCollection> singleTauJets;
    iEvent.getByLabel( singleTau, singleTauJets );
    EMIsolatedTauTagInfoCollection::const_iterator i = singleTauJets->begin();
    for(;i !=singleTauJets->end(); i++ ) {
      double discriminator = (*i).discriminator();
      if(discriminator > 0) {
	//	cout <<"Crasha qui?2"<<endl;
	const CaloJet* pippo = dynamic_cast<const CaloJet*>(&(i->jet()));
	jetIsolatedCollection->push_back(*pippo );
      }else{
	const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(i->jet()));
	jetNotIsolatedCollection->push_back(*notPippo );
      }
    }
    
    //All DoubleTau jets are put into the non isolated collection
    edm::Handle<EMIsolatedTauTagInfoCollection> doubleTauJets;
    iEvent.getByLabel( doubleTau, doubleTauJets );
    //    cout <<"Crasha qui?2.5"<<endl;
    EMIsolatedTauTagInfoCollection::const_iterator ii = doubleTauJets->begin();
    for(;ii !=doubleTauJets->end(); ii++ ) {
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
      }else{
	const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(i->jet()));
	jetNotIsolatedCollection->push_back(*notPippo );
      }
    }
    edm::Handle<EMIsolatedTauTagInfoCollection> doubleTauJets;
    iEvent.getByLabel( doubleTau, doubleTauJets );
    EMIsolatedTauTagInfoCollection::const_iterator ii = doubleTauJets->begin();
    for(;ii !=doubleTauJets->end(); ii++ ) {
      double discriminator = (*ii).discriminator();
      if(discriminator > 0) {
	const CaloJet* pippo = dynamic_cast<const CaloJet*>(&(ii->jet()));
	jetIsolatedCollection->push_back(*pippo );
      }else{
	const CaloJet* notPippo = dynamic_cast<const CaloJet*>(&(ii->jet()));
	jetNotIsolatedCollection->push_back(*notPippo );
      }
    }
    
  }
  //  cout <<"Crasha qui?3"<<endl;
  auto_ptr<reco::CaloJetCollection> isolatedTaus(jetIsolatedCollection);
  auto_ptr<reco::CaloJetCollection> notIsolatedTaus(jetNotIsolatedCollection);
  //  cout <<"Crasha qui?3.5"<<endl;
  iEvent.put(isolatedTaus, "Isolated");
  //cout <<"Crasha qui?4"<<endl;
  iEvent.put(notIsolatedTaus,"NotIsolated");
  
}
