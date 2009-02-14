#include "DQM/HLTEvF/interface/HLTTauDQMCaloPlotter.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

HLTTauDQMCaloPlotter::HLTTauDQMCaloPlotter(const edm::ParameterSet& iConfig,int etbins,int etabins,int phibins,double maxpt,bool ref,double dr):
 l2TauInfoAssoc_(iConfig.getParameter<edm::InputTag>("L2InfoAssociationInput")),
 doRef_(ref),
 matchDeltaRMC_(dr),
 triggerTag_((iConfig.getParameter<std::string>("DQMFolder"))),
 l2Isolated_(iConfig.getParameter<edm::InputTag>("L2IsolatedJets")),
 EtMax_(maxpt),
 NPtBins_(etbins),
 NEtaBins_(etabins),
 NPhiBins_(phibins)
{
   store = &*edm::Service<DQMStore>();
 
  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      jetEt= store->book1D("L2TauEt","L2 #tau E_{t}",NPtBins_,0,EtMax_);
      jetEta= store->book1D("L2TauEta","L2 #tau #eta",NEtaBins_,-2.5,2.5);
      jetPhi= store->book1D("L2TauPhi","L2 #tau #phi",NPhiBins_,-3.2,3.2);

      isoJetEt= store->book1D("L2IsoTauEt","L2 isolated #tau E_{t}",NPtBins_,0,EtMax_);
      isoJetEta= store->book1D("L2IsoTauEta","L2 isolated #tau #eta",NEtaBins_,-2.5,2.5);
      isoJetPhi= store->book1D("L2IsoTauPhi","L2 isolated #tau #phi",NPhiBins_,-3.2,3.2);

      ecalIsolEt=store->book1D("L2EcalIsolation","ECAL Isolation",40,0,20);
      hcalIsolEt=store->book1D("L2HcalIsolation","HCAL Isolation",40,0,20);
      seedHcalEt=store->book1D("L2HighestHCALCluster","Highest HCAL Cluster",40,0,80);
      seedEcalEt=store->book1D("L2HighestECALCluster","Highest ECAL Cluster",25,0,50);
      nEcalClusters=store->book1D("L2NEcalClusters","Nucmber of ECAL Clusters",20,0,20);
      ecalClusterEtaRMS=store->book1D("L2EcalEtaRMS","ECAL Cluster #eta RMS",15,0,0.05);
      ecalClusterPhiRMS=store->book1D("L2EcalPhiRMS","ECAL Cluster #phi RMS",30,0,0.1);
      ecalClusterDeltaRRMS=store->book1D("L2EcalDeltaRRMS","ECAL Cluster #DeltaR RMS",30,0,0.1);
      nHcalClusters=store->book1D("L2NHcalClusters","Nucmber of HCAL Clusters",20,0,20);
      hcalClusterEtaRMS=store->book1D("L2HcalEtaRMS","HCAL Cluster #eta RMS",15,0,0.05);
      hcalClusterPhiRMS=store->book1D("L2HcalPhiRMS","HCAL Cluster #phi RMS",30,0,0.1);
      hcalClusterDeltaRRMS=store->book1D("L2HcalDeltaRRMS","HCAL Cluster #DeltaR RMS",30,0,0.1);


      //      clusterDeltaRRMS->getTH1F()->Sumw2();

      EtEffNum=store->book1D("L2TauEtEffNum","Efficiency vs E_{t}(Numerator)",NPtBins_,0,EtMax_);
      EtEffNum->getTH1F()->Sumw2();

      EtEffDenom=store->book1D("L2TauEtEffDenom","Efficiency vs E_{t}(Denominator)",NPtBins_,0,EtMax_);
      EtEffDenom->getTH1F()->Sumw2();

      EtaEffNum=store->book1D("L2TauEtaEffNum","Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
      EtaEffNum->getTH1F()->Sumw2();

      EtaEffDenom=store->book1D("L2TauEtaEffDenom","Efficiency vs #eta(Denominator)",NEtaBins_,-2.5,2.5);
      EtaEffDenom->getTH1F()->Sumw2();

      PhiEffNum=store->book1D("L2TauPhiEffNum","Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
      PhiEffNum->getTH1F()->Sumw2();

      PhiEffDenom=store->book1D("L2TauPhiEffDenom","Efficiency vs #phi(Denominator)",NPhiBins_,-3.2,3.2);
      PhiEffDenom->getTH1F()->Sumw2();
    }
 
}

HLTTauDQMCaloPlotter::~HLTTauDQMCaloPlotter()
{
}

void
HLTTauDQMCaloPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,const LVColl& McInfo)
{
   using namespace edm;
   using namespace reco;

   edm::Handle<L2TauInfoAssociation> l2TauInfoAssoc;
   edm::Handle<CaloJetCollection> l2Isolated;

   if(iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc))//get the Association class
     {
       //If the Collection exists do work
       if(l2TauInfoAssoc->size()>0)
	 for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
       
	     //Retrieve the Jet From the AssociationMap
	     const Jet& jet =*(p->key);
	     
	     std::pair<bool,LV> m =match(jet,McInfo); 
	     
	     if((doRef_&&m.first)||(!doRef_))
		 {
		   ecalIsolEt->Fill(l2info.ecalIsolEt());
		   hcalIsolEt->Fill(l2info.hcalIsolEt());
		   seedEcalEt->Fill(l2info.seedEcalHitEt());
		   seedHcalEt->Fill(l2info.seedHcalHitEt());

		   nEcalClusters->Fill(l2info.nEcalHits());
		   ecalClusterEtaRMS->Fill(l2info.ecalClusterShape()[0]);
		   ecalClusterPhiRMS->Fill(l2info.ecalClusterShape()[1]);
		   ecalClusterDeltaRRMS->Fill(l2info.ecalClusterShape()[2]);

		   nHcalClusters->Fill(l2info.nHcalHits());
		   hcalClusterEtaRMS->Fill(l2info.hcalClusterShape()[0]);
		   hcalClusterPhiRMS->Fill(l2info.hcalClusterShape()[1]);
		   hcalClusterDeltaRRMS->Fill(l2info.hcalClusterShape()[2]);

		   jetEt->Fill(jet.et());
		   jetEta->Fill(jet.eta());
		   jetPhi->Fill(jet.phi());

		   LV refLV;
		   if(doRef_)
		     refLV = m.second;
		   else
		     refLV = jet.p4();

		   EtEffDenom->Fill(refLV.pt());
		   EtaEffDenom->Fill(refLV.eta());
		   PhiEffDenom->Fill(refLV.phi());

  		   if(iEvent.getByLabel(l2Isolated_,l2Isolated))
		     if(l2Isolated.isValid());
		     {
		       if(matchJet(jet,*l2Isolated))
			 { 
	   		   isoJetEt->Fill(jet.et());
			   isoJetEta->Fill(jet.eta());
			   isoJetPhi->Fill(jet.phi());

			   EtEffNum->Fill(refLV.pt());
			   EtaEffNum->Fill(refLV.eta());
			   PhiEffNum->Fill(refLV.phi());
			 }
		     }
		 }
	   } 
	       
     }

}




std::pair<bool,LV> 
HLTTauDQMCaloPlotter::match(const reco::Jet& jet,const LVColl& McInfo)
{



  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 LV mLV;

 if(McInfo.size()>0)
  for(LVColl::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4(),*it);
	  if(delta<matchDeltaRMC_)
	    {
	      matched=true;
	      mLV=*it;
	    }
   }


 std::pair<bool,LV> p = std::make_pair(matched,mLV);
 return p;
}

bool 
HLTTauDQMCaloPlotter::matchJet(const reco::Jet& jet,const reco::CaloJetCollection& McInfo)
{
  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;

 if(McInfo.size()>0)
  for(reco::CaloJetCollection::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4(),it->p4());
	  if(delta<matchDeltaRMC_)
	    {
	      matched=true;
    
	    }
   }
 return matched;
}


