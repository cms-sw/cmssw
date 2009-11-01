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
 CaloJetCollection myTmpJets;
 myTmpJets.clear();

 int iL1Jet = 0;
 for( vtag::const_iterator s = jetSrc.begin(); s != jetSrc.end(); ++ s ) {
   edm::Handle<CaloJetCollection> tauJets;
   iEvent.getByLabel( * s, tauJets );
   for(CaloJetCollection::const_iterator iTau = tauJets->begin();iTau !=tauJets->end();iTau++)
     { 
     //Create a Map to associate to every Jet its L1SeedId, i.e. 0,1,2 or 3
     if(iTau->et() > mEt_Min)
       //       myL2L1JetsMap.insert(pair<int, const CaloJet>(iL1Jet, *(iTau)));
       myTmpJets.push_back(*(iTau));
     }
   iL1Jet++;
 }

 auto_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection); 
 //Removing the collinear jets

 for(unsigned int iTau1 = 0; iTau1 <  myTmpJets.size();iTau1++)
     { 
       for(unsigned int iTau2 = iTau1; iTau2 <  myTmpJets.size();iTau2++)
	 { 
	   double deltaR = ROOT::Math::VectorUtil::DeltaR(myTmpJets[iTau1].p4().Vect(), myTmpJets[iTau2].p4().Vect());
	   if(deltaR < 0.1) tauL2jets->push_back(myTmpJets[iTau1]);
	 }
     }
//    cout <<"Size of L2 jets "<<tauL2jets->size()<<endl;

  iEvent.put(tauL2jets);

}
