#include "RecoTauTag/HLTProducers/interface/L1HLTTauMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"

//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
using namespace l1extra;

L1HLTTauMatching::L1HLTTauMatching(const edm::ParameterSet& iConfig)
{
  jetSrc = consumes<PFTauCollection>(iConfig.getParameter<InputTag>("JetSrc") );
  tauTrigger = consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1TauTrigger") );
  mEt_Min = iConfig.getParameter<double>("EtMin");
  
  produces<PFTauCollection>();
}
L1HLTTauMatching::~L1HLTTauMatching(){ }

void L1HLTTauMatching::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
	
  using namespace edm;
  using namespace std;
  using namespace reco;
  using namespace trigger;
  using namespace l1extra;
	
  auto_ptr<PFTauCollection> tauL2jets(new PFTauCollection);
	
  double deltaR = 1.0;
  double matchingR = 0.5;
  //Getting HLT jets to be matched
  edm::Handle<PFTauCollection > tauJets;
  iEvent.getByToken( jetSrc, tauJets );

  //		std::cout <<"Size of input jet collection "<<tauJets->size()<<std::endl;
		
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
  iEvent.getByToken(tauTrigger,l1TriggeredTaus);
		
		
  tauCandRefVec.clear();
  jetCandRefVec.clear();
  
  l1TriggeredTaus->getObjects( trigger::TriggerL1TauJet,tauCandRefVec);
  l1TriggeredTaus->getObjects( trigger::TriggerL1CenJet,jetCandRefVec);
  math::XYZPoint a(0.,0.,0.);
  CaloJet::Specific f;
		
  for( unsigned int iL1Tau=0; iL1Tau <tauCandRefVec.size();iL1Tau++)
    {  
      for(unsigned int iJet=0;iJet<tauJets->size();iJet++)
	{
	  //Find the relative L2TauJets, to see if it has been reconstructed
	  const PFTau &  myJet = (*tauJets)[iJet];
	  deltaR = ROOT::Math::VectorUtil::DeltaR(myJet.p4().Vect(), (tauCandRefVec[iL1Tau]->p4()).Vect());
	  if(deltaR < matchingR ) {
	    //		 LeafCandidate myLC(myJet);
	    if(myJet.leadPFChargedHadrCand().isNonnull()){
	      a =  myJet.leadPFChargedHadrCand()->vertex();  
	    }
	    PFTau myPFTau(std::numeric_limits<int>::quiet_NaN(), myJet.p4(), a);
	    if(myJet.pt() > mEt_Min) {
	      //		  tauL2LC->push_back(myLC);
	      tauL2jets->push_back(myPFTau);
	    }
	    break;
	  }
	}
    }  
  
  for(unsigned int iL1Tau=0; iL1Tau <jetCandRefVec.size();iL1Tau++)
    {  
      for(unsigned int iJet=0;iJet<tauJets->size();iJet++)
	{
	  const PFTau &  myJet = (*tauJets)[iJet];
	  //Find the relative L2TauJets, to see if it has been reconstructed
	  deltaR = ROOT::Math::VectorUtil::DeltaR(myJet.p4().Vect(), (jetCandRefVec[iL1Tau]->p4()).Vect());
	  if(deltaR < matchingR ) {
	    //		 LeafCandidate myLC(myJet);
	    if(myJet.leadPFChargedHadrCand().isNonnull()){
	      a =  myJet.leadPFChargedHadrCand()->vertex();  
	    }
            
	    PFTau myPFTau(std::numeric_limits<int>::quiet_NaN(), myJet.p4(),a);
	    if(myJet.pt() > mEt_Min) {
	      //tauL2LC->push_back(myLC);
	      tauL2jets->push_back(myPFTau);
	    }
	    break;
	  }
	}
    }
  

  //std::cout <<"Size of L1HLT matched jets "<<tauL2jets->size()<<std::endl;

  iEvent.put(tauL2jets);
  // iEvent.put(tauL2LC);
}
