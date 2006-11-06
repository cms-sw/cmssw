#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"

//
// class decleration
//
using namespace reco;
using namespace edm;


L2TauJetsProvider::L2TauJetsProvider(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<InputTag>("JetSrc");

  produces<CaloJetCollection>("SingleTau");
  produces<CaloJetCollection>("DoubleTau");
  produces<CaloJetCollection>("LeptonTau");
  
}

L2TauJetsProvider::~L2TauJetsProvider(){ }

void L2TauJetsProvider::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

 using namespace edm;
 using namespace std;
 using namespace reco;
 Handle<CaloJetCollection> jets;
 iEvent.getByLabel(jetSrc,jets);
 CaloJetCollection* singleTauTmp = new CaloJetCollection();
 CaloJetCollection* doubleTauTmp= new CaloJetCollection();
 CaloJetCollection* leptonTauTmp= new CaloJetCollection();

	mEt_SingleTau = 30.;
	mEt_DoubleTau = 66.;
	mEt_CombTau = 50.;
	  
  //Loop over the jetSrc to split the jets
  //Et cut will be replaced with L1 Trigger bit

  CaloJetCollection::const_iterator iJet = jets->begin();
  
  for(;iJet!=jets->end();iJet++)
    {
      if(iJet->pt() >= mEt_SingleTau ) singleTauTmp->push_back(*(iJet));
  
    }
  
  iJet= jets->begin();
  for(;iJet!=jets->end();iJet++)
    {
      //Fill doubleTauJet collection if the jet is above L1 Double Thresholds
      if(iJet->pt() >= mEt_DoubleTau && iJet->pt() < mEt_SingleTau) doubleTauTmp->push_back(*(iJet));

      //Fill doubleTauJet collection if the jet is below L1 Double Thresholds but there is at least one Tau triggering single Tau
      if(singleTauTmp->size() > 0 && iJet->pt() < mEt_DoubleTau && iJet->pt() >= mEt_CombTau) doubleTauTmp->push_back(*(iJet));

      //Otherwise fill Lepton+Tau collection
      if(singleTauTmp->size() == 0 && iJet->pt() < mEt_DoubleTau && iJet->pt() >= mEt_CombTau) leptonTauTmp->push_back(*(iJet));

    }
  std::auto_ptr<CaloJetCollection> singleTaujets(singleTauTmp);
  std::auto_ptr<CaloJetCollection> doubleTaujets(doubleTauTmp);
  std::auto_ptr<CaloJetCollection> leptonTaujets(leptonTauTmp);
  cout <<"Size of SingleTau "<<singleTauTmp->size()<<endl;
  cout <<"Size of DoubleTau "<<doubleTauTmp->size()<<endl;

  iEvent.put(singleTaujets, "SingleTau");
  iEvent.put(doubleTaujets, "DoubleTau");
  iEvent.put(leptonTaujets, "LeptonTau");

}
