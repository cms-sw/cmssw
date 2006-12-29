#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
#include "Math/GenVector/VectorUtil.h"

//
// class decleration
//
using namespace reco;
using namespace edm;


L2TauJetsProvider::L2TauJetsProvider(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  l1ParticleMap = iConfig.getParameter<InputTag>("L1ParticleMap");
  mEt_ExtraTau = iConfig.getParameter<double>("EtExtraTau");
  mEt_ExtraTau = iConfig.getParameter<double>("EtLeptonTau");
  
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


 //Getting all the L1Seeds
 Handle< L1JetParticleCollection > tauColl ; 
 InputTag tauJetInputTag( l1ParticleMap.label(), "Tau" ) ;
 iEvent.getByLabel( tauJetInputTag, tauColl );
 const L1JetParticleCollection & myL1Tau  = *(tauColl.product()); 

 //Getting the Collections of L2ReconstructedJets from L1Seeds
 myL2L1JetsMap.clear();
  int iL1Jet = 0;
 for( vtag::const_iterator s = jetSrc.begin(); s != jetSrc.end(); ++ s ) {
   edm::Handle<CaloJetCollection> tauJets;
   iEvent.getByLabel( * s, tauJets );
   CaloJetCollection::const_iterator iTau = tauJets->begin();
   if(iTau != tauJets->end()){
     //Create a Map to associate to every Jet its L1SeedId, i.e. 0,1,2 or 3
     myL2L1JetsMap.insert(pair<int, const CaloJet>(iL1Jet, *(iTau)));
   }
   iL1Jet++;
 }


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
 cout <<"Trigger SingleTau "<<singleTauFired<<endl;
 cout <<"Trigger DoubleTau "<<doubleTauFired<<endl;
 cout <<"SingleTau objects: "<<myL1SingleTaus.size()<<endl;
 cout <<"DoubleTau objects: "<<myL1DoubleTaus.size()<<endl;

 //Loop over the jetSrc to split the jets
  

  double deltaR;
  double matchingR = 0.01;
  int iJet=0;
  //Loop over the Map to find which jets has fired the trigger
  //myL1Tau is the Collection of L1TauCandidates (from 0 to max  4 elements)
  for(;iJet<myL1Tau.size();iJet++)
    {
      bool alreadyMatched = false;
      //Find the relative L2TauJets, to see if it has been reconstructed
      map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
      if(myL2itr!=myL2L1JetsMap.end()){
	L1JetParticleVectorRef::const_iterator myTau1 = myL1SingleTaus.begin();
	//	cout <<"*** Loop over L1 SingleTau "<<endl; 
	for(;myTau1 != myL1SingleTaus.end();myTau1++)
	  {
	    //Calculate the DeltaR between L1TauCandidate and L1Tau which fired the trigger
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myTau1)->p4().Vect());
	    //	    cout <<"SingleTau "<<(*myTau1)->pt() << " "<<(*myTau1)->eta() << " " << (*myTau1)->phi() <<endl;
	    //	    cout <<"deltaR "<<deltaR<<endl;
	    if(deltaR < matchingR) {
	      //Getting back from the map the L2TauJet
	      const CaloJet myL2TauJet = myL2itr->second;
	      singleTauTmp->push_back(myL2TauJet);
	      alreadyMatched = true;
	      break;
	    }
	  }  
	
	if(alreadyMatched) continue;
	L1JetParticleVectorRef::const_iterator myTau2 = myL1DoubleTaus.begin();
	//	cout <<"*** Loop over L1 DoubleTau "<<endl; 
	for(;myTau2 != myL1DoubleTaus.end();myTau2++)
	  {
	    //	    cout <<"DoubleTau "<<(*myTau2)->pt() << " "<<(*myTau2)->eta() << " " << (*myTau2)->phi() <<endl;
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myTau2)->p4().Vect());
	    //	    cout <<"detaR "<<deltaR<<endl;
	    if(deltaR < matchingR) {
	      const CaloJet myL2TauJet = myL2itr->second;
	      doubleTauTmp->push_back(myL2TauJet);
	      alreadyMatched = true;
	      break;
	    }
	  }  
	if(alreadyMatched) continue;
	
	double etL1Cand = myL1Tau[iJet].et();
	if(singleTauFired && (!doubleTauFired) && etL1Cand > mEt_ExtraTau ){
	  const CaloJet myL2TauJet = myL2itr->second;
	  doubleTauTmp->push_back(myL2TauJet);
	  alreadyMatched = true;
	}
	if(alreadyMatched) continue;
	if(etL1Cand > mEt_LeptonTau){
	  const CaloJet myL2TauJet = myL2itr->second;
	  leptonTauTmp->push_back(myL2TauJet);
	}	  
      }
    }



  auto_ptr<CaloJetCollection> singleTaujets(singleTauTmp);
  auto_ptr<CaloJetCollection> doubleTaujets(doubleTauTmp);
  auto_ptr<CaloJetCollection> leptonTaujets(leptonTauTmp);

  cout <<"Size of SingleTau "<<singleTauTmp->size()<<endl;
  cout <<"Size of DoubleTau "<<doubleTauTmp->size()<<endl;
  cout <<"Size of LeptonTau "<<leptonTauTmp->size()<<endl;

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
      if(singleTauSize > 1 && doubleTauSize >0) l1Decision =1;
    }
  }
  if(!singleTauFired){
    if(doubleTauFired) l1Decision =3;
  }

  cout <<"Tau Trigger internal code "<<l1Decision <<endl;

  auto_ptr< vector<int> > myL1BookKeep(new vector<int>);
  myL1BookKeep->push_back(l1Decision);
  
  
  
  iEvent.put(singleTaujets, "SingleTau");
  iEvent.put(doubleTaujets, "DoubleTau");
  iEvent.put(leptonTaujets, "LeptonTau");
  iEvent.put(myL1BookKeep);

}
