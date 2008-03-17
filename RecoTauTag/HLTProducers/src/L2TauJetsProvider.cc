#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

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
  l1Particles = iConfig.getParameter<InputTag>("L1Particles");
  tauTrigger = iConfig.getParameter<InputTag>("L1TauTrigger");
  mEt_Min = iConfig.getParameter<double>("EtMin");

  
  
  produces<CaloJetCollection>();
}

L2TauJetsProvider::~L2TauJetsProvider(){ }

void L2TauJetsProvider::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

 using namespace edm;
 using namespace std;
 using namespace reco;
 using namespace trigger;
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
 
  auto_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection);
  

 //Loop over the jetSrc to select the proper jets
  

  double deltaR;
  double matchingR = 0.01;
  //Loop over the Map to find which jets has fired the trigger
  //myL1Tau is the Collection of L1TauCandidates (from 0 to max  4 elements)
  //get the list of trigger candidates from the HLTL1SeedGT filter

  Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;

  if(iEvent.getByLabel(tauTrigger,l1TriggeredTaus)){

    //    typedef std::vector<l1extra::L1JetParticleRef>     VRl1jet;
vector<L1JetParticleRef> tauCandRefVec;
    L1JetParticleRef tauCandRef;
    l1TriggeredTaus->getObjects(trigger::TriggerL1TauJet,tauCandRefVec);


    for( unsigned int iL1Tau=0; iL1Tau <tauCandRefVec.size();iL1Tau++)
      {  
	tauCandRef = tauCandRefVec[iL1Tau];
	for(int iJet=0;iJet<myL1Tau.size();iJet++)
	  {
	    bool alreadyMatched = false;
	    //Find the relative L2TauJets, to see if it has been reconstructed
	    map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
	    if(myL2itr!=myL2L1JetsMap.end()){
	      
	    //Calculate the DeltaR between L1TauCandidate and L1Tau which fired the trigger
	    deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Tau[iJet].p4().Vect(), (tauCandRef->p4()).Vect());
	    if(deltaR < matchingR) {
	      //Getting back from the map the L2TauJet
	      const CaloJet myL2TauJet = myL2itr->second;
	      if(myL2TauJet.pt() > mEt_Min){
		tauL2jets->push_back(myL2TauJet);
		break;
	      }
	    }
	  }

	}
    }

  }
  iEvent.put(tauL2jets);

}
