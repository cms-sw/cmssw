#include "DQMOffline/Trigger/interface/HLTTauCaloDQMOfflineSource.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

HLTTauCaloDQMOfflineSource::HLTTauCaloDQMOfflineSource(const edm::ParameterSet& iConfig):
 l2TauInfoAssoc_(iConfig.getParameter<edm::InputTag>("L2InfoAssociationInput")),
 mcColl_(iConfig.getParameter<edm::InputTag>("refCollection")),
 met_(iConfig.getParameter<edm::InputTag >("MET")),
 doRef_(iConfig.getParameter<bool>("doReference")),
 matchDeltaRMC_(iConfig.getParameter<double>("MatchDeltaR")),
 triggerTag_((iConfig.getParameter<std::string>("DQMFolder"))),
 l2Isolated_(iConfig.getParameter<edm::InputTag>("L2IsolatedJets")),
 outFile_(iConfig.getParameter<std::string>("OutputFileName")),
 EtMin_(iConfig.getParameter<double>("EtMin")),
 EtMax_(iConfig.getParameter<double>("EtMax")),
 NBins_(iConfig.getParameter<int>("NBins"))

{

   store = &*edm::Service<DQMStore>();
  
  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      jetEt= store->book1D("L2tauCandEt","tauCandEt",NBins_,EtMin_,EtMax_);
      jetEta= store->book1D("L2tauCandEta","tauCandEta",50,-2.5,2.5);
      jetPhi= store->book1D("L2tauCandPhi","tauCandPhi",63,-3.14,3.14);
      ecalIsolEt=store->book1D("L2ecalIsolEt","ecalIsolEt",40,0,20);
      towerIsolEt=store->book1D("L2towerIsolEt","towerIsolEt",40,0,20);
      seedTowerEt=store->book1D("L2seedTowerEt","seedTowerEt",40,0,80);
      nClusters=store->book1D("L2nClusters","nClusters",20,0,20);
      clusterEtaRMS=store->book1D("L2clusterEtaRMS","clusterEtaRMS",25,0,0.5);
      clusterPhiRMS=store->book1D("L2clusterPhiRMS","clusterPhiRMS",25,0,0.5);
      clusterDeltaRRMS=store->book1D("L2clusterDeltaRRMS","clusterDeltaRRMS",25,0,0.5);
      EtEffNum=store->book1D("L2EtEffNum","Efficiency vs E_{t}(Numerator)",NBins_,EtMin_,EtMax_);
      EtEffDenom=store->book1D("L2EtEffDenom","Efficiency vs E_{t}(Denominator)",NBins_,EtMin_,EtMax_);
      MET=store->book1D("MET","Missing E_{t}",NBins_,EtMin_,EtMax_);
      
    }
 
}


HLTTauCaloDQMOfflineSource::~HLTTauCaloDQMOfflineSource()
{
}



void
HLTTauCaloDQMOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association)
   iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc);

   Handle<LVColl> McInfo; //Handle To The Truth!!!!
   iEvent.getByLabel(mcColl_,McInfo);

   Handle<reco::CaloJetCollection> l2Isolated;
   iEvent.getByLabel(l2Isolated_,l2Isolated);

   if (!l2TauInfoAssoc.isValid() || !l2Isolated.isValid())
     {
      edm::LogInfo("HLTTauCaloDQMOfflineSource") << "l2TauInfoAssoc object not found, "
      "skipping event"; 
    return;
     }


   std::vector<l1extra::L1JetParticleRef> tauCandRefVec;

   if(l2TauInfoAssoc.isValid())//get the Association class
     {

       //Lets see if we have MC w matching or real data
       if(McInfo.isValid())
       //If the Collection exists do work
       if(l2TauInfoAssoc->size()>0)
	 for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
       
	     //Retrieve the Jet From the AssociationMap
	     const Jet& jet =*(p->key);
  
	     if((doRef_&&match(jet,*McInfo))||(!doRef_))
		 {
		   ecalIsolEt->Fill(l2info.ecalIsolEt());
		   towerIsolEt->Fill(l2info.hcalIsolEt());
		   nClusters->Fill(l2info.nEcalHits());
		   seedTowerEt->Fill(l2info.seedHcalHitEt());
		   clusterEtaRMS->Fill(l2info.ecalClusterShape()[0]);
		   clusterPhiRMS->Fill(l2info.ecalClusterShape()[1]);
		   clusterDeltaRRMS->Fill(l2info.ecalClusterShape()[2]);
		   jetEt->Fill(jet.et());
		   jetEta->Fill(jet.eta());
		   jetPhi->Fill(jet.phi());
	      
		   EtEffDenom->Fill(jet.et());

  		   if(l2Isolated.isValid())
		     {
		       if(matchJet(jet,*l2Isolated)) 
			    EtEffNum->Fill(jet.et());

		     }

		 }

	   
	   } 
	       
     }

   //Plot the missing Et. To be used in SingleTau mainly
   Handle<CaloMETCollection> met; iEvent.getByLabel(met_,met);
   if(met.isValid())//get the Association class
     {
       MET->Fill((*met)[0].pt());
     }


}



void 
HLTTauCaloDQMOfflineSource::beginJob(const edm::EventSetup&)
{

}


void 
HLTTauCaloDQMOfflineSource::endJob() {
 
  //Get Efficiency

  EtEffNum->getTH1F()->Sumw2();
  EtEffDenom->getTH1F()->Sumw2();

  //Write file
  if(outFile_.size()>0)
    if (store) store->save (outFile_);

}

bool 
HLTTauCaloDQMOfflineSource::match(const reco::Jet& jet,const LVColl& McInfo)
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
HLTTauCaloDQMOfflineSource::matchJet(const reco::Jet& jet,const reco::CaloJetCollection& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;

 if(McInfo.size()>0)
  for(reco::CaloJetCollection::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),it->p4().Vect());
	  if(delta<matchDeltaRMC_)
	    {
	      matched=true;	     
	    }
   }



 return matched;
}


