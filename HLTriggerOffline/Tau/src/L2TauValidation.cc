#include "HLTriggerOffline/Tau/interface/L2TauValidation.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

L2TauValidation::L2TauValidation(const edm::ParameterSet& iConfig):
 l2TauInfoAssoc_(iConfig.getParameter<edm::InputTag>("L2InfoAssociationInput")),
 mcColl_(iConfig.getParameter<edm::InputTag>("MatchedCollection")),
 l1taus_(iConfig.getParameter<edm::InputTag>("L1TauTrigger")),
 matchLevel_(iConfig.getParameter<int>("MatchLevel")),
 matchDeltaRMC_(iConfig.getParameter<double>("MatchDeltaRMC")),
 matchDeltaRL1_(iConfig.getParameter<double>("MatchDeltaRL1")),
 outFile_(iConfig.getParameter<std::string>("OutputFileName"))
{

  DQMStore* store = &*edm::Service<DQMStore>();
  
  if(store)
    {
      //Create the histograms
      store->setCurrentFolder("HLTTauL2"+l2TauInfoAssoc_.label());

      jetEt= store->book1D("tauCandEt","tauCandEt",50,0,100);
      jetEta= store->book1D("tauCandEta","tauCandEta",50,-2.5,2.5);
      jetPhi= store->book1D("tauCandPhi","tauCandPhi",63,-3.14,3.14);
      ecalIsolEt=store->book1D("ecalIsolEt","ecalIsolEt",40,0,20);
      towerIsolEt=store->book1D("towerIsolEt","towerIsolEt",40,0,20);
      seedTowerEt=store->book1D("seedTowerEt","seedTowerEt",40,0,80);
      nClusters=store->book1D("nClusters","nClusters",20,0,20);
      clusterEtaRMS=store->book1D("clusterEtaRMS","clusterEtaRMS",25,0,0.5);
      clusterPhiRMS=store->book1D("clusterPhiRMS","clusterPhiRMS",25,0,0.5);
      clusterDeltaRRMS=store->book1D("clusterDeltaRRMS","clusterDeltaRRMS",25,0,0.5);


    }
  

 
}


L2TauValidation::~L2TauValidation()
{
}



void
L2TauValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association)
   Handle<LVColl> McInfo; //Handle To The Truth!!!!


   Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus; //Handle to the L1


   std::vector<l1extra::L1JetParticleRef> tauCandRefVec;

   iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc);//get the Association class
 

    //Lets see if we have MC w matching or real data
    if(matchLevel_>0) //get MC Match
      {
	iEvent.getByLabel(mcColl_,McInfo);
      }
    if(matchLevel_>1) //get L1 Objects
      {
	iEvent.getByLabel(l1taus_,l1TriggeredTaus);
        l1TriggeredTaus->getObjects(trigger::TriggerL1TauJet,tauCandRefVec);
  
      }
    




       //If the Collection exists do work
    if(l2TauInfoAssoc->size()>0)
     for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p)
     {
       //Retrieve The L2TauIsolationInfo Class from the AssociationMap
       const L2TauIsolationInfo l2info = p->val;
       
        //Retrieve the Jet From the AssociationMap
       const Jet& jet =*(p->key);

  
      
	  if((matchLevel_>0&&match(jet,*McInfo))||matchLevel_==0)
	    if((matchLevel_>1&&matchL1(jet,tauCandRefVec))||matchLevel_<2)
	    {
	      ecalIsolEt->Fill(l2info.ECALIsolConeCut);
	      towerIsolEt->Fill(l2info.TowerIsolConeCut);
	      nClusters->Fill(l2info.ECALClusterNClusters);
	      seedTowerEt->Fill(l2info.SeedTowerEt);
	      clusterEtaRMS->Fill(l2info.ECALClusterEtaRMS);
	      clusterPhiRMS->Fill(l2info.ECALClusterPhiRMS);
	      clusterDeltaRRMS->Fill(l2info.ECALClusterDRRMS);
	      jetEt->Fill(jet.et());
	      jetEta->Fill(jet.eta());
	      jetPhi->Fill(jet.phi());
	    }

	   
     } 
	       
     



}



void 
L2TauValidation::beginJob(const edm::EventSetup&)
{

}


void 
L2TauValidation::endJob() {
 
  //Write file
if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outFile_);

}

bool 
L2TauValidation::match(const reco::Jet& jet,const LVColl& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;

 if(McInfo.size()>0)
  for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),*it);
	  if(delta<matchDeltaRMC_)
	    {
	      matched=true;
	     
	    }
   }



 return matched;
}





bool 
L2TauValidation::matchL1(const reco::Jet& jet,std::vector<l1extra::L1JetParticleRef>& tauCandRefVec)
{

  bool match = false;

  if(tauCandRefVec.size()>0)
    for(unsigned int iL1Tau=0; iL1Tau <tauCandRefVec.size();iL1Tau++)
	  {  
	    double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),tauCandRefVec[iL1Tau]->p4().Vect());
	    if(delta<matchDeltaRL1_)
	      match=true;
	     }

  return match;
}



