#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
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

L2TauJetsProvider::L2TauJetsProvider(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  for( vtag::const_iterator s = jetSrc.begin(); s != jetSrc.end(); ++ s ) {
    edm::EDGetTokenT<CaloJetCollection> aToken = consumes<CaloJetCollection>( *s );
    jetSrcToken.push_back(aToken);
  }
  l1ParticlesTau = consumes<L1JetParticleCollection>(iConfig.getParameter<InputTag>("L1ParticlesTau"));
  l1ParticlesJet = consumes<L1JetParticleCollection>(iConfig.getParameter<InputTag>("L1ParticlesJet"));
  tauTrigger = consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1TauTrigger"));
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

 
 //Getting the Collections of L2ReconstructedJets from L1Seeds
 //and removing the collinear jets
 myL2L1JetsMap.clear();
 int iL1Jet = 0;
 typedef std::vector<edm::EDGetTokenT<reco::CaloJetCollection> > vtag_token;
 for( vtag_token::const_iterator s = jetSrcToken.begin(); s != jetSrcToken.end(); ++ s ) {
   edm::Handle<CaloJetCollection> tauJets;
   iEvent.getByToken( * s, tauJets );
   CaloJetCollection::const_iterator iTau = tauJets->begin();
   if(iTau != tauJets->end()){
     //Create a Map to associate to every Jet its L1SeedId, i.e. 0,1,2 or 3
     myL2L1JetsMap.insert(std::pair<int, const CaloJet>(iL1Jet, *(iTau)));
   }
   iL1Jet++;
 }
 std::auto_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection);

 //Loop over the jetSrc to select the proper jets
  
  double deltaR = 0.1;
  double matchingR = 0.01;
  //Loop over the Map to find which jets has fired the trigger
  //myL1Tau is the Collection of L1TauCandidates (from 0 to max  4 elements)
  //get the list of trigger candidates from the HLTL1SeedGT filter
  edm::Handle< L1JetParticleCollection > tauColl ; 
  edm::Handle< L1JetParticleCollection > jetColl ; 
  
  iEvent.getByToken(l1ParticlesTau, tauColl);
  iEvent.getByToken(l1ParticlesJet, jetColl);

  const L1JetParticleCollection  myL1Tau = *(tauColl.product());

  const L1JetParticleCollection  myL1Jet = *(jetColl.product());  

    L1JetParticleCollection myL1Obj;
    myL1Obj.reserve(8);
    
    for(unsigned int i=0;i<myL1Tau.size();i++)
      {
	myL1Obj.push_back(myL1Tau[i]);
      }
    for(unsigned int j=0;j<myL1Jet.size();j++)
      {
	myL1Obj.push_back(myL1Jet[j]);
      }


    edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
    if(iEvent.getByToken(tauTrigger,l1TriggeredTaus)){
    
      
      tauCandRefVec.clear();
      jetCandRefVec.clear();
      
      l1TriggeredTaus->getObjects( trigger::TriggerL1TauJet,tauCandRefVec);
      l1TriggeredTaus->getObjects( trigger::TriggerL1CenJet,jetCandRefVec);
      
      for( unsigned int iL1Tau=0; iL1Tau <tauCandRefVec.size();iL1Tau++)
	{  
	  for(unsigned int iJet=0;iJet<myL1Obj.size();iJet++)
	    {
	      //Find the relative L2TauJets, to see if it has been reconstructed
	    std::map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
	    if(myL2itr!=myL2L1JetsMap.end()){
	      //Calculate the DeltaR between L1TauCandidate and L1Tau which fired the trigger
	      if(&tauCandRefVec[iL1Tau]) 
		deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Obj[iJet].p4().Vect(), (tauCandRefVec[iL1Tau]->p4()).Vect());
	      if(deltaR < matchingR ) {
	      //	      Getting back from the map the L2TauJet
		const CaloJet myL2TauJet = myL2itr->second;
		if(myL2TauJet.pt() > mEt_Min) tauL2jets->push_back(myL2TauJet);
		myL2L1JetsMap.erase(myL2itr->first);
		break;
		
	      }
	    }
	    
	  }
      }
      
    for(unsigned int iL1Tau=0; iL1Tau <jetCandRefVec.size();iL1Tau++)
      {  
	for(unsigned int iJet=0;iJet<myL1Obj.size();iJet++)
	  {
	    //Find the relative L2TauJets, to see if it has been reconstructed
	    std::map<int, const reco::CaloJet>::const_iterator myL2itr = myL2L1JetsMap.find(iJet);
	    if(myL2itr!=myL2L1JetsMap.end()){
	      //Calculate the DeltaR between L1TauCandidate and L1Tau which fired the trigger
	      if(&jetCandRefVec[iL1Tau])
	        deltaR = ROOT::Math::VectorUtil::DeltaR(myL1Obj[iJet].p4().Vect(), (jetCandRefVec[iL1Tau]->p4()).Vect());
	      if(deltaR < matchingR) {
		// Getting back from the map the L2TauJet
		const CaloJet myL2TauJet = myL2itr->second;
		
		if(myL2TauJet.pt() > mEt_Min) tauL2jets->push_back(myL2TauJet);
		myL2L1JetsMap.erase(myL2itr->first);
		break;
		
	      }
	    }
	    
	  }
      }
    
  }
  //  std::cout <<"Size of L2 jets "<<tauL2jets->size()<<std::endl;

  iEvent.put(tauL2jets);

}
