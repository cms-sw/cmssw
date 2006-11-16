#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
#include "Math/GenVector/VectorUtil.h"

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
  iEvent.getByLabel( l1ParticleMap, mapColl );

  const L1ParticleMap& singleTauMap = ( *mapColl )[L1ParticleMap::kSingleTau ] ;
  const L1ParticleMap& doubleTauMap = ( *mapColl )[L1ParticleMap::kDoubleTau ] ;
  const L1JetParticleVectorRef& myL1SingleTaus = singleTauMap.jetParticles();
  const L1JetParticleVectorRef& myL1DoubleTaus = doubleTauMap.jetParticles();

  bool singleTauFired = singleTauMap.triggerDecision() ;
  bool doubleTauFired = doubleTauMap.triggerDecision() ;
  cout <<"SingleTau objects: "<<myL1SingleTaus.size()<<endl;
  cout <<"DoubleTau objects: "<<myL1DoubleTaus.size()<<endl;

  Handle< L1JetParticleCollection > tauColl ; 
  InputTag tauJetInputTag( l1ParticleMap.label(), "Tau" ) ;

  iEvent.getByLabel( tauJetInputTag, tauColl );


	  
  //Loop over the jetSrc to split the jets
  
  CaloJetCollection::const_iterator iJet = jets->begin();
  L1JetParticleVectorRef::const_iterator myTau1 = myL1SingleTaus.begin();
  L1JetParticleVectorRef::const_iterator myTau2 = myL1DoubleTaus.begin();
  L1JetParticleCollection::const_iterator tauItr = tauColl->begin();

  //FIXME to be put in the cfg file
  mEt_CombTau = 50.;
  double deltaR;
  double matchingR = 0.3;
  for(;iJet!=jets->end();iJet++)
    {
      bool alreadyMatched = false;
      //      cout <<"L2 TauJets  "<< iJet->pt()<< " "<<iJet->eta()<<" "<<iJet->phi()<<endl;
      for(;myTau1 != myL1SingleTaus.end();myTau1++)
	{
	  deltaR = ROOT::Math::VectorUtil::DeltaR(iJet->p4().Vect(), (*myTau1)->p4().Vect());
	  //	  cout <<"SingleTau "<<(*myTau1)->pt() << " "<<(*myTau1)->eta() << " " << (*myTau1)->phi() <<endl;
	  if(deltaR < matchingR) {
	    singleTauTmp->push_back(*(iJet));
	    alreadyMatched = true;
	  }
	}  
      if(alreadyMatched) continue;
      for(;myTau2 != myL1DoubleTaus.end();myTau2++)
	{
	  //	  cout <<"DoubleTau "<<(*myTau2)->pt() << " "<<(*myTau2)->eta() << " " << (*myTau2)->phi() <<endl;
	  deltaR = ROOT::Math::VectorUtil::DeltaR(iJet->p4().Vect(), (*myTau2)->p4().Vect());
	  if(deltaR < matchingR) {
	    doubleTauTmp->push_back(*(iJet));
	    alreadyMatched = true;
	  }
	}  
      if(alreadyMatched) continue;
      for(;tauItr != tauColl->end() ;++tauItr )
	{
	  //	  cout <<"LeptonTau "<<tauItr->pt() << " "<<tauItr->eta() << " " << tauItr->phi() <<endl;
	  deltaR = ROOT::Math::VectorUtil::DeltaR(iJet->p4().Vect(), tauItr->p4().Vect());
	  if(deltaR < matchingR){
	    if((!doubleTauFired) && tauItr->pt() > mEt_CombTau ){
	      doubleTauTmp->push_back(*(iJet));
	    }else{
	      leptonTauTmp->push_back(*(iJet));
	    }
	  
	  }
	}
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
