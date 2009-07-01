#include "RecoTauTag/HLTProducers/interface/L2TauJetsMerger.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
using namespace l1extra;

L2TauJetsMerger::L2TauJetsMerger(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  //l1ParticlesTau = iConfig.getParameter<InputTag>("L1ParticlesTau");
  //l1ParticlesJet = iConfig.getParameter<InputTag>("L1ParticlesJet");
  //tauTrigger = iConfig.getParameter<InputTag>("L1TauTrigger");
  mEt_Min = iConfig.getParameter<double>("EtMin");
  
  produces<CaloJetCollection>();
}

L2TauJetsMerger::~L2TauJetsMerger(){ }

void L2TauJetsMerger::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

 using namespace edm;
 using namespace std;
 using namespace reco;

 //Getting all the L1Seeds

 
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
     if(iTau->et() > mEt_Min)
       myL2L1JetsMap.insert(pair<int, const CaloJet>(iL1Jet, *(iTau)));
   }
   iL1Jet++;
 }

 auto_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection); 
 //Removing the collinear jets
 for(int iJet =0;iJet<iL1Jet;iJet++)
   {
     map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
     if(myL2itr!=myL2L1JetsMap.end()){
       
       const CaloJet my1stJet = myL2itr->second;
       for(int i2Jet = iJet+1;i2Jet<iL1Jet;i2Jet++)
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
  for(int iJet =0;iJet<iL1Jet;iJet++)
   {
     map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
     if(myL2itr!=myL2L1JetsMap.end()){
       
       const CaloJet my1stJet = myL2itr->second;
       tauL2jets->push_back(my1stJet);
     }
   }

  
  //  cout <<"Size of L2 jets "<<tauL2jets->size()<<endl;

  iEvent.put(tauL2jets);

}
