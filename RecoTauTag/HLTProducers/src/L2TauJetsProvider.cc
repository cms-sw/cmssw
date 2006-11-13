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
  produces< std::vector<int> >();
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
  auto_ptr<CaloJetCollection> singleTaujets(singleTauTmp);
  auto_ptr<CaloJetCollection> doubleTaujets(doubleTauTmp);
  auto_ptr<CaloJetCollection> leptonTaujets(leptonTauTmp);

  cout <<"Size of SingleTau "<<singleTauTmp->size()<<endl;
  cout <<"Size of DoubleTau "<<doubleTauTmp->size()<<endl;

   int l1Decision =0;
  int singleTauSize = singleTauTmp->size();
  int doubleTauSize = doubleTauTmp->size();


  //This have to be modified taking L1 bits


  if(singleTauSize < 2 && doubleTauSize == 0)    l1Decision = 1; //Pure SingleTau Trigger
  if(singleTauSize >1 && doubleTauSize == 0)     l1Decision = 2;//Single && DoubleTau
  if(singleTauSize >1 && doubleTauSize > 0) l1Decision = 3;//Single && DoubleTau      
  if(singleTauSize == 1 && doubleTauSize > 0) l1Decision = 4;//Single && DoubleTau, To check the mixed case 
  if(singleTauSize == 0 && doubleTauSize >1) l1Decision = 6;//Pure Double
  
  //There is missing the mixed case!!!!!!!!

  auto_ptr< vector<int> > myL1BookKeep(new vector<int>);
  myL1BookKeep->push_back(l1Decision);
  
  
  
  iEvent.put(singleTaujets, "SingleTau");
  iEvent.put(doubleTaujets, "DoubleTau");
  iEvent.put(leptonTaujets, "LeptonTau");
  iEvent.put(myL1BookKeep);

}
