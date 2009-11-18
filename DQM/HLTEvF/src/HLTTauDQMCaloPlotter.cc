#include "DQM/HLTEvF/interface/HLTTauDQMCaloPlotter.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

HLTTauDQMCaloPlotter::HLTTauDQMCaloPlotter(const edm::ParameterSet& iConfig,int etbins,int etabins,int phibins,double maxpt,bool ref,double dr):
 l2preJets_(iConfig.getParameter<std::vector<edm::InputTag> >("L2RegionalJets")),
 l2TauInfoAssoc_(iConfig.getParameter<edm::InputTag>("L2InfoAssociationInput")),
 doRef_(ref),
 matchDeltaRMC_(dr),
 triggerTag_(iConfig.getParameter<std::string>("DQMFolder")),
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

      preJetEt= store->book1D("L2PreTauEt","L2 regional #tau E_{t}",NPtBins_,0,EtMax_);
       preJetEt->getTH1F()->GetXaxis()->SetTitle("L2 regional Jet E_{T}");
       preJetEt->getTH1F()->GetYaxis()->SetTitle("entries");

    preJetEta= store->book1D("L2PreTauEta","L2 regional #tau #eta",NEtaBins_,-2.5,2.5);
       preJetEta->getTH1F()->GetXaxis()->SetTitle("L2 regional Jet #eta");
       preJetEta->getTH1F()->GetYaxis()->SetTitle("entries");

    preJetPhi= store->book1D("L2PreTauPhi","L2 regional #tau #phi",NPhiBins_,-3.2,3.2);
       preJetPhi->getTH1F()->GetXaxis()->SetTitle("L2 regional Jet #phi");
       preJetPhi->getTH1F()->GetYaxis()->SetTitle("entries");

    jetEt= store->book1D("L2TauEt","L2 #tau E_{t}",NPtBins_,0,EtMax_);
       jetEt->getTH1F()->GetXaxis()->SetTitle("L2 selected Jet E_{T}");
       jetEt->getTH1F()->GetYaxis()->SetTitle("entries");

    jetEta= store->book1D("L2TauEta","L2 #tau #eta",NEtaBins_,-2.5,2.5);
       jetEta->getTH1F()->GetXaxis()->SetTitle("L2 selected Jet #eta");
       jetEta->getTH1F()->GetYaxis()->SetTitle("entries");

    jetPhi= store->book1D("L2TauPhi","L2 #tau #phi",NPhiBins_,-3.2,3.2);
       jetPhi->getTH1F()->GetXaxis()->SetTitle("L2 selected Jet #phi");
       jetPhi->getTH1F()->GetYaxis()->SetTitle("entries");

    jetEtRes= store->book1D("L2TauEtResol","L2 #tau E_{t} resolution",40,-2,2);
       jetEtRes->getTH1F()->GetXaxis()->SetTitle("L2 selected Jet #phi");
       jetEtRes->getTH1F()->GetYaxis()->SetTitle("entries");

    isoJetEt= store->book1D("L2IsoTauEt","L2 isolated #tau E_{t}",NPtBins_,0,EtMax_);
       isoJetEt->getTH1F()->GetXaxis()->SetTitle("L2 isolated Jet E_{T}");
       isoJetEt->getTH1F()->GetYaxis()->SetTitle("entries");

    isoJetEta= store->book1D("L2IsoTauEta","L2 isolated #tau #eta",NEtaBins_,-2.5,2.5);
       isoJetEta->getTH1F()->GetXaxis()->SetTitle("L2 isolated Jet #eta");
       isoJetEta->getTH1F()->GetYaxis()->SetTitle("entries");

    isoJetPhi= store->book1D("L2IsoTauPhi","L2 isolated #tau #phi",NPhiBins_,-3.2,3.2);
       isoJetPhi->getTH1F()->GetXaxis()->SetTitle("L2 isolated Jet #phi");
       isoJetPhi->getTH1F()->GetYaxis()->SetTitle("entries");

    ecalIsolEt=store->book1D("L2EcalIsolation","ECAL Isolation",40,0,20);
       ecalIsolEt->getTH1F()->GetXaxis()->SetTitle("L2 ECAL isolation E_{T}");
       ecalIsolEt->getTH1F()->GetYaxis()->SetTitle("entries");

    hcalIsolEt=store->book1D("L2HcalIsolation","HCAL Isolation",40,0,20);
       hcalIsolEt->getTH1F()->GetXaxis()->SetTitle("L2 HCAL isolation E_{T}");
       hcalIsolEt->getTH1F()->GetYaxis()->SetTitle("entries");

    seedHcalEt=store->book1D("L2HighestHCALCluster","Highest HCAL Cluster",40,0,80);
       seedHcalEt->getTH1F()->GetXaxis()->SetTitle("HCAL seed  E_{T}");
       seedHcalEt->getTH1F()->GetYaxis()->SetTitle("entries");

    seedEcalEt=store->book1D("L2HighestECALCluster","Highest ECAL Cluster",25,0,50);
       seedEcalEt->getTH1F()->GetXaxis()->SetTitle("ECAL seed  E_{T}");
       seedEcalEt->getTH1F()->GetYaxis()->SetTitle("entries");

    nEcalClusters=store->book1D("L2NEcalClusters","Nucmber of ECAL Clusters",20,0,20);
       nEcalClusters->getTH1F()->GetXaxis()->SetTitle("n. of ECAL Clusters");
       nEcalClusters->getTH1F()->GetYaxis()->SetTitle("entries");

    ecalClusterEtaRMS=store->book1D("L2EcalEtaRMS","ECAL Cluster #eta RMS",15,0,0.05);
       ecalClusterEtaRMS->getTH1F()->GetXaxis()->SetTitle("ECAL cluster #eta RMS");
       ecalClusterEtaRMS->getTH1F()->GetYaxis()->SetTitle("entries");

    ecalClusterPhiRMS=store->book1D("L2EcalPhiRMS","ECAL Cluster #phi RMS",30,0,0.1);
       ecalClusterPhiRMS->getTH1F()->GetXaxis()->SetTitle("ECAL cluster #phi RMS");
       ecalClusterPhiRMS->getTH1F()->GetYaxis()->SetTitle("entries");


    ecalClusterDeltaRRMS=store->book1D("L2EcalDeltaRRMS","ECAL Cluster #DeltaR RMS",30,0,0.1);
       ecalClusterDeltaRRMS->getTH1F()->GetXaxis()->SetTitle("ECAL cluster #DeltaR RMS");
       ecalClusterDeltaRRMS->getTH1F()->GetYaxis()->SetTitle("entries");

    nHcalClusters=store->book1D("L2NHcalClusters","Nucmber of HCAL Clusters",20,0,20);
       nHcalClusters->getTH1F()->GetXaxis()->SetTitle("n. of ECAL Clusters");
       nHcalClusters->getTH1F()->GetYaxis()->SetTitle("entries");

      hcalClusterEtaRMS=store->book1D("L2HcalEtaRMS","HCAL Cluster #eta RMS",15,0,0.05);
       hcalClusterEtaRMS->getTH1F()->GetXaxis()->SetTitle("HCAL cluster #eta RMS");
       hcalClusterEtaRMS->getTH1F()->GetYaxis()->SetTitle("entries");

      hcalClusterPhiRMS=store->book1D("L2HcalPhiRMS","HCAL Cluster #phi RMS",30,0,0.1);
       hcalClusterPhiRMS->getTH1F()->GetXaxis()->SetTitle("HCAL cluster #phi RMS");
       hcalClusterPhiRMS->getTH1F()->GetYaxis()->SetTitle("entries");

      hcalClusterDeltaRRMS=store->book1D("L2HcalDeltaRRMS","HCAL Cluster #DeltaR RMS",30,0,0.1);
       hcalClusterDeltaRRMS->getTH1F()->GetXaxis()->SetTitle("HCAL cluster #DeltaR RMS");
       hcalClusterDeltaRRMS->getTH1F()->GetYaxis()->SetTitle("entries");


      store->setCurrentFolder(triggerTag_+"/EfficiencyHelpers");

      recoEtEffNum=store->book1D("L2RecoTauEtEffNum","Efficiency vs E_{t}(Numerator)",NPtBins_,0,EtMax_);
      recoEtEffNum->getTH1F()->Sumw2();

      recoEtEffDenom=store->book1D("L2RecoTauEtEffDenom","Efficiency vs E_{t}(Denominator)",NPtBins_,0,EtMax_);
      recoEtEffDenom->getTH1F()->Sumw2();

      recoEtaEffNum=store->book1D("L2RecoTauEtaEffNum","Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
      recoEtaEffNum->getTH1F()->Sumw2();

      recoEtaEffDenom=store->book1D("L2RecoTauEtaEffDenom","Efficiency vs #eta(Denominator)",NEtaBins_,-2.5,2.5);
      recoEtaEffDenom->getTH1F()->Sumw2();

      recoPhiEffNum=store->book1D("L2RecoTauPhiEffNum","Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
      recoPhiEffNum->getTH1F()->Sumw2();

      recoPhiEffDenom=store->book1D("L2RecoTauPhiEffDenom","Efficiency vs #phi(Denominator)",NPhiBins_,-3.2,3.2);
      recoPhiEffDenom->getTH1F()->Sumw2();

      isoEtEffNum=store->book1D("L2IsoTauEtEffNum","Efficiency vs E_{t}(Numerator)",NPtBins_,0,EtMax_);
      isoEtEffNum->getTH1F()->Sumw2();

      isoEtEffDenom=store->book1D("L2IsoTauEtEffDenom","Efficiency vs E_{t}(Denominator)",NPtBins_,0,EtMax_);
      isoEtEffDenom->getTH1F()->Sumw2();

      isoEtaEffNum=store->book1D("L2IsoTauEtaEffNum","Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
      isoEtaEffNum->getTH1F()->Sumw2();

      isoEtaEffDenom=store->book1D("L2IsoTauEtaEffDenom","Efficiency vs #eta(Denominator)",NEtaBins_,-2.5,2.5);
      isoEtaEffDenom->getTH1F()->Sumw2();

      isoPhiEffNum=store->book1D("L2IsoTauPhiEffNum","Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
      isoPhiEffNum->getTH1F()->Sumw2();

      isoPhiEffDenom=store->book1D("L2IsoTauPhiEffDenom","Efficiency vs #phi(Denominator)",NPhiBins_,-3.2,3.2);
      isoPhiEffDenom->getTH1F()->Sumw2();
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
   edm::Handle<CaloJetCollection> l2Regional;
   CaloJetCollection l2RegionalJets;

   //Fill the regional Jets
   //Be carefu;l with the double counting

   for(unsigned int j=0;j<l2preJets_.size();++j) {

     bool gotPreJets =true;
     try {
       iEvent.getByLabel(l2preJets_[j],l2Regional);
     }
     catch (cms::Exception& exception) {
       gotPreJets =false;
     }


     if(gotPreJets)
       if((!l2Regional.failedToGet()))
       {
       for(unsigned int i=0;i<l2Regional->size();++i)
	 {
	   const Jet& jet = (*l2Regional)[i];
	   if(jet.pt()>10)
	     {
	       std::pair<bool,LV> m =match(jet,McInfo); 

	       if((doRef_&&m.first)||(!doRef_))
		 {
		   preJetEt->Fill(jet.et());
		   preJetEta->Fill(jet.eta());
		   preJetPhi->Fill(jet.phi());

		   LV refLV;
		   if(doRef_)
		     refLV = m.second;
		   else
		     refLV = jet.p4();
		   
		   recoEtEffDenom->Fill(refLV.pt());
		   recoEtaEffDenom->Fill(refLV.eta());
		   recoPhiEffDenom->Fill(refLV.phi());
	           l2RegionalJets.push_back((*l2Regional)[i]);
		 }
	     }

	 }
     }
   }


     bool gotL2 =true;
     try {
       iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc);
     }
     catch (cms::Exception& exception) {
       gotL2 =false;
     }


     if(gotL2)
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
		   
		   if(doRef_)
		     jetEtRes->Fill((jet.pt()-refLV.pt())/refLV.pt());

		   if(matchJet(jet,l2RegionalJets))
		     {
		       recoEtEffNum->Fill(refLV.pt());
		       recoEtaEffNum->Fill(refLV.eta());
		       recoPhiEffNum->Fill(refLV.phi());
		     }

		   isoEtEffDenom->Fill(refLV.pt());
		   isoEtaEffDenom->Fill(refLV.eta());
		   isoPhiEffDenom->Fill(refLV.phi());

  		   if(iEvent.getByLabel(l2Isolated_,l2Isolated))
		     if(l2Isolated.isValid());
		     {
		       if(matchJet(jet,*l2Isolated))
			 { 
	   		   isoJetEt->Fill(jet.et());
			   isoJetEta->Fill(jet.eta());
			   isoJetPhi->Fill(jet.phi());

			   isoEtEffNum->Fill(refLV.pt());
			   isoEtaEffNum->Fill(refLV.eta());
			   isoPhiEffNum->Fill(refLV.phi());
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




