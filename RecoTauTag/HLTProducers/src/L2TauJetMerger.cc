#include "RecoTauTag/HLTProducers/interface/L2TauJetMerger.h"

//
// class decleration
//

L2TauJetMerger::L2TauJetMerger(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");

  produces<reco::CaloJetCollection>();
}

L2TauJetMerger::~L2TauJetMerger(){ }

void L2TauJetMerger::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

 using namespace edm;
 using namespace std;
 using namespace reco;
 auto_ptr<CaloJetCollection> mergedTaus(new CaloJetCollection);
 for( vtag::const_iterator s = jetSrc.begin(); s != jetSrc.end(); ++ s ) {
   edm::Handle<CaloJetCollection> tauJets;
   iEvent.getByLabel( * s, tauJets );
   CaloJetCollection::const_iterator iTau = tauJets->begin();
   if(iTau != tauJets->end())
     mergedTaus->push_back(*iTau);
 }

 cout <<"Number of Selected Taus "<<mergedTaus->size()<<endl;
  iEvent.put(mergedTaus);
}
