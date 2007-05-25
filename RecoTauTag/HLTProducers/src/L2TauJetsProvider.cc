#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
#include "Math/GenVector/VectorUtil.h"

//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
using namespace l1extra;

L2TauJetsProvider::L2TauJetsProvider(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  l1ParticleMap = iConfig.getParameter<InputTag>("L1ParticleMap");
  l1Particles = iConfig.getParameter<InputTag>("L1Particles");
  singleTauTrigger = iConfig.getParameter<string>("L1SingleTauTrigger");
  singleTauMETTrigger = iConfig.getParameter<string>("L1SingleTauMETTrigger");
  doubleTauTrigger = iConfig.getParameter<string>("L1DoubleTauTrigger");
  electronTauTrigger = iConfig.getParameter<string>("L1IsoEMTauTrigger");
  muonTauTrigger = iConfig.getParameter<string>("L1MuonTrigger");
  mEt_Min = iConfig.getParameter<double>("EtMin");
  mEt_ExtraTau = iConfig.getParameter<double>("EtExtraTau");
  
  
  produces<CaloJetCollection>("SingleTau");
  produces<CaloJetCollection>("SingleTauMET");
  produces<CaloJetCollection>("DoubleTau");
  produces<CaloJetCollection>("DoubleExtraTau");
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


 iEvent.getByLabel( l1Particles, tauColl );
 const L1JetParticleCollection & myL1Tau  = *(tauColl.product()); 

 //Getting the Collections of L2ReconstructedJets from L1Seeds
 //and removing the collinear jets
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

 //Removing the collinear jets
 for(int iJet =0;iJet<4;iJet++)
   {
     map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
     if(myL2itr!=myL2L1JetsMap.end()){

       const CaloJet my1stJet = myL2itr->second;
       for(int i2Jet = iJet+1;i2Jet<4;i2Jet++)
	 {
	   map<int, const reco::CaloJet>::const_iterator my2L2itr = myL2L1JetsMap.find(i2Jet);
	   if(my2L2itr!=myL2L1JetsMap.end()){
	     const CaloJet my2ndJet = my2L2itr->second;
	   
	     double deltaR = ROOT::Math::VectorUtil::DeltaR(my1stJet.p4().Vect(), my2ndJet.p4().Vect());
	     if(deltaR < 0.1) 
	       {
		 //		 cout <<"Collinear jets "<<deltaR<<endl;
		 myL2L1JetsMap.erase(my2L2itr->first);
	       }
	   }
	 }
     }
   }

  auto_ptr<CaloJetCollection> singleTaujets(new CaloJetCollection);
  auto_ptr<CaloJetCollection> singleTauMETjets(new CaloJetCollection);
  auto_ptr<CaloJetCollection> doubleTaujets(new CaloJetCollection);
  auto_ptr<CaloJetCollection> doubleExtraTaujets(new CaloJetCollection);
  auto_ptr<CaloJetCollection> leptonTaujets(new CaloJetCollection);




 Handle< L1ParticleMapCollection > mapColl ;
 iEvent.getByLabel( l1ParticleMap, mapColl );
 unsigned int singleTauTrigger_ =(unsigned int) (L1ParticleMap::triggerType(singleTauTrigger));

 unsigned int singleTauMETTrigger_ =(unsigned int) (L1ParticleMap::triggerType(singleTauMETTrigger));

 unsigned int doubleTauTrigger_ =(unsigned int)(L1ParticleMap::triggerType(doubleTauTrigger));

 unsigned int electronTauTrigger_ =(unsigned int)(L1ParticleMap::triggerType(electronTauTrigger));

 unsigned int muonTauTrigger_ =(unsigned int)(L1ParticleMap::triggerType(muonTauTrigger));

 if(singleTauTrigger_ < 0 || singleTauTrigger_  > L1ParticleMap::kNumOfL1TriggerTypes-1) 
   throw edm::Exception(edm::errors::Configuration);

 if(singleTauMETTrigger_ < 0 || singleTauMETTrigger_  > L1ParticleMap::kNumOfL1TriggerTypes-1) 
   throw edm::Exception(edm::errors::Configuration);

 if(doubleTauTrigger_ < 0 || doubleTauTrigger_  > L1ParticleMap::kNumOfL1TriggerTypes-1) 
   throw edm::Exception(edm::errors::Configuration);

 if(electronTauTrigger_ < 0 || electronTauTrigger_  > L1ParticleMap::kNumOfL1TriggerTypes-1) 
   throw edm::Exception(edm::errors::Configuration);

 if(muonTauTrigger_ < 0 || muonTauTrigger_  > L1ParticleMap::kNumOfL1TriggerTypes-1) 
   throw edm::Exception(edm::errors::Configuration);



 const L1ParticleMap& singleTauMap = ( *mapColl )[singleTauTrigger_] ;
 const L1ParticleMap& singleTauMETMap = ( *mapColl )[singleTauMETTrigger_] ;
 const L1ParticleMap& doubleTauMap = ( *mapColl )[doubleTauTrigger_] ;
 const L1ParticleMap& electronTauMap = ( *mapColl )[electronTauTrigger_] ;
 const L1ParticleMap& muonTauMap = ( *mapColl )[muonTauTrigger_] ;
 


 const L1JetParticleVectorRef& myL1SingleTaus = singleTauMap.jetParticles();
 const L1JetParticleVectorRef& myL1SingleTausMET = singleTauMETMap.jetParticles();
 const L1JetParticleVectorRef& myL1DoubleTaus = doubleTauMap.jetParticles();
 const L1JetParticleVectorRef& myL1ElectronTaus = electronTauMap.jetParticles();
 const L1JetParticleVectorRef& myL1MuonTaus = muonTauMap.jetParticles();

 bool singleTauFired = singleTauMap.triggerDecision() ;
 bool doubleTauFired = doubleTauMap.triggerDecision() ;

 /*
 cout <<"Trigger SingleTau "<<singleTauFired<<endl;
 cout <<"Trigger DoubleTau "<<doubleTauFired<<endl;
 cout <<"SingleTau objects: "<<myL1SingleTaus.size()<<endl;
 cout <<"DoubleTau objects: "<<myL1DoubleTaus.size()<<endl;
 */

 

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


	//SingleTau
	L1JetParticleVectorRef::const_iterator myTau1 = myL1SingleTaus.begin();
	for(;myTau1 != myL1SingleTaus.end();myTau1++)
	  {
	    //Calculate the DeltaR between L1TauCandidate and L1Tau which fired the trigger
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myTau1)->p4().Vect());
	    if(deltaR < matchingR) {
	      //Getting back from the map the L2TauJet
	      const CaloJet myL2TauJet = myL2itr->second;
	      if(myL2TauJet.pt() > mEt_Min){
		singleTaujets->push_back(myL2TauJet);
		alreadyMatched = true;
		break;
	      }
	    }
	  }  
	
	//SingleTauMET
	L1JetParticleVectorRef::const_iterator myTau1MET = myL1SingleTausMET.begin();
	for(;myTau1MET != myL1SingleTausMET.end();myTau1MET++)
	  {
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myTau1MET)->p4().Vect());
	    if(deltaR < matchingR) {
	      const CaloJet myL2TauJet = myL2itr->second;
	      if(myL2TauJet.pt() > mEt_Min){
		singleTauMETjets->push_back(myL2TauJet);
		//	alreadyMatched = true;
	      }
	    }
	  } 


	//DoubleTau
	if(alreadyMatched) continue;
	L1JetParticleVectorRef::const_iterator myTau2 = myL1DoubleTaus.begin();
	for(;myTau2 != myL1DoubleTaus.end();myTau2++)
	  {
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myTau2)->p4().Vect());
	    if(deltaR < matchingR) {
	      const CaloJet myL2TauJet = myL2itr->second;
	      if(myL2TauJet.pt() > mEt_Min){
		doubleTaujets->push_back(myL2TauJet);
		alreadyMatched = true;
		break;
	      }
	    }
	  }  
	if(alreadyMatched) continue;
	
	double etL1Cand = myL1Tau[iJet].et(); 
	if(singleTauFired && (!doubleTauFired) && etL1Cand > mEt_ExtraTau ){
	  const CaloJet myL2TauJet = myL2itr->second;
	  if(myL2TauJet.pt() > mEt_Min){
	    doubleExtraTaujets->push_back(myL2TauJet);
	  }
	}



	L1JetParticleVectorRef::const_iterator myElectronTau = myL1ElectronTaus.begin();
	for(;myElectronTau != myL1ElectronTaus.end();myElectronTau++)
	  {
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myElectronTau)->p4().Vect());
	    if(deltaR < matchingR) {
	      const CaloJet myL2TauJet = myL2itr->second;
	      if(myL2TauJet.pt() > mEt_Min){
		leptonTaujets->push_back(myL2TauJet);
		alreadyMatched = true;
		break;
	      }
	    }
	  }  
	if(alreadyMatched) continue;

	L1JetParticleVectorRef::const_iterator myMuonTau = myL1MuonTaus.begin();
	for(;myMuonTau != myL1MuonTaus.end();myMuonTau++)
	  {
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (*myMuonTau)->p4().Vect());
	    if(deltaR < matchingR) {
	      const CaloJet myL2TauJet = myL2itr->second;
	      if(myL2TauJet.pt() > mEt_Min){
		leptonTaujets->push_back(myL2TauJet);
		alreadyMatched = true;
		break;
	      }
	    }
	  }  
	if(alreadyMatched) continue;

      }
    }



  /*  
  cout <<"Size of SingleTau "<<singleTaujets->size()<<endl;
  cout <<"Size of DoubleTau "<<doubleTaujets->size()<<endl;
  cout <<"Size of LeptonTau "<<leptonTaujets->size()<<endl;
  */



  int l1Decision =0;
  int singleTauSize = singleTaujets->size();
  int doubleTauSize = doubleTaujets->size();



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

  //  cout <<"Tau Trigger internal code "<<l1Decision <<endl;

  auto_ptr< vector<int> > myL1BookKeep(new vector<int>);
  myL1BookKeep->push_back(l1Decision);
  
  
  
  iEvent.put(singleTaujets, "SingleTau");
  iEvent.put(singleTauMETjets, "SingleTauMET"); 
  iEvent.put(doubleTaujets, "DoubleTau");
  iEvent.put(doubleExtraTaujets, "DoubleExtraTau");
  iEvent.put(leptonTaujets, "LeptonTau");
  iEvent.put(myL1BookKeep);

}
