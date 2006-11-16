#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"

//
// class decleration
//
using namespace reco;
using namespace edm;


L2TauJetsProvider::L2TauJetsProvider(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<InputTag>("JetSrc");
  l1ParticleMap = iConfig.getParameter<InputTag>("L1ParticleMap");

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
 using namespace l1extra;
 Handle<CaloJetCollection> jets;
 iEvent.getByLabel(jetSrc,jets);
 CaloJetCollection* singleTauTmp = new CaloJetCollection();
 CaloJetCollection* doubleTauTmp= new CaloJetCollection();
 CaloJetCollection* leptonTauTmp= new CaloJetCollection();

  Handle< L1ParticleMapCollection > mapColl ;
  iEvent.getByLabel( l1ParticleMap, mapColl ) ;
  const L1ParticleMap& singleTauMap = ( *mapColl )[L1ParticleMap::kSingleTau ] ;
  const L1ParticleMap& doubleTauMap = ( *mapColl )[L1ParticleMap::kDoubleTau ] ;

  bool singleTauFired = singleTauMap.triggerDecision() ;
  bool doubleTauFired = doubleTauMap.triggerDecision() ;


  //FIXEM:
  //To be fixed with L1 energy and thresholds
  //TO BE CAREFULL VERY DANGEROUS AND ERROR PRONE
  mEt_SingleTau = 30.;
  mEt_DoubleTau = 12.;
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
      if(doubleTauFired){
	if(iJet->pt() >= mEt_DoubleTau && iJet->pt() < mEt_SingleTau) doubleTauTmp->push_back(*(iJet));
      }else{
      //Fill doubleTauJet collection if the jet is below L1 Double Thresholds but there is at least one Tau triggering single Tau
	if(singleTauFired && iJet->pt() < mEt_DoubleTau && iJet->pt() >= mEt_CombTau) doubleTauTmp->push_back(*(iJet));
      }
	//Otherwise fill Lepton+Tau collection
      if(iJet->pt() < mEt_DoubleTau && iJet->pt() >= mEt_CombTau) leptonTauTmp->push_back(*(iJet));
    }
  auto_ptr<CaloJetCollection> singleTaujets(singleTauTmp);
  auto_ptr<CaloJetCollection> doubleTaujets(doubleTauTmp);
  auto_ptr<CaloJetCollection> leptonTaujets(leptonTauTmp);

  cout <<"Size of SingleTau "<<singleTauTmp->size()<<endl;
  cout <<"Size of DoubleTau "<<doubleTauTmp->size()<<endl;

  int l1Decision =0;
  int singleTauSize = singleTauTmp->size();
  int doubleTauSize = doubleTauTmp->size();



  //Coding decisions:
  // 0: No doubleTau is possible
  // 1: L1 singleTau candidate has  to be confirmed in at least 1 jet
  // 2: L1 singleTau candidate has  to be confirmed in at least 2 jets
  // 3: L1 singleTau has NOT to be confirmed
  


  if(singleTauFired){
    if(!doubleTauFired) {
      if(doubleTauSize ==0) l1Decision =0;
      if(doubleTauSize>0) l1Decision =1; //Mixed Tau case
    }else{
      //Single and DoubleTau fired
      if(doubleTauSize == 0) l1Decision=2;
      if(singleTauSize == 1 && doubleTauSize ==1) l1Decision =1;
      if(doubleTauSize >1) l1Decision =3;
    }
  }
  if(!singleTauFired){
    l1Decision =3;
  }


  auto_ptr< vector<int> > myL1BookKeep(new vector<int>);
  myL1BookKeep->push_back(l1Decision);
  
  
  
  iEvent.put(singleTaujets, "SingleTau");
  iEvent.put(doubleTaujets, "DoubleTau");
  iEvent.put(leptonTaujets, "LeptonTau");
  iEvent.put(myL1BookKeep);

}
