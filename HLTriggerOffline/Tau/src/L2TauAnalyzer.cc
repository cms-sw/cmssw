#include "HLTriggerOffline/Tau/interface/L2TauAnalyzer.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

L2TauAnalyzer::L2TauAnalyzer(const edm::ParameterSet& iConfig):
 l2TauInfoAssoc_(iConfig.getParameter<edm::InputTag>("L2InfoAssociationInput")),
 rootFile_(iConfig.getParameter<std::string>("outputFileName")),
 IsSignal_(iConfig.getParameter<bool>("IsSignal")),
 mcColl_(iConfig.getParameter<edm::InputTag>("MatchedCollection")),
 genJets_(iConfig.getParameter<edm::InputTag>("GenJetCollection")), 
 l1taus_(iConfig.getParameter<edm::InputTag>("L1TauTrigger"))

{
  //File Setup
  l2file = new TFile(rootFile_.c_str(),"recreate");

  //Tree Setup
  l2tree = new TTree("l2tree","Level 2 Tau Tree");

  // int cl_Nclusters,triggerBit,matchBit;
  //   double  ecalIsol_Et,towerIsol_Et,cl_etaRMS,cl_phiRMS,cl_drRMS,eta,phi,et; 


  //Initialize the vars
  ecalIsol_Et=0.;
  towerIsol_Et=0.;
  cl_etaRMS=0.;
  cl_phiRMS=0.;
  cl_drRMS=0.;
  MCeta=0.;
  MCet=0.;
  cl_Nclusters=0;
  seedTowerEt = 0.;
  matchBit=0;
  matchL1Bit=0;
  JetEt=0.;

  //Setup Branches
  l2tree->Branch("ecalIsol_Et",&ecalIsol_Et,"ecalIsol_Et/F");
  l2tree->Branch("towerIsol_Et",&towerIsol_Et,"towerIsol_Et/F");
  l2tree->Branch("cl_etaRMS",&cl_etaRMS,"cl_etaRMS/F");
  l2tree->Branch("cl_phiRMS",&cl_phiRMS,"cl_phiRMS/F");
  l2tree->Branch("cl_drRMS",&cl_drRMS,"cl_drRMS/F");
  l2tree->Branch("MCeta",&MCeta,"MCeta/F");
  l2tree->Branch("MCet",&MCet,"MCet/F");
  l2tree->Branch("cl_Nclusters",&cl_Nclusters,"cl_Nclusters/I");
  l2tree->Branch("MCMatched",&matchBit,"matchBit/I");
  l2tree->Branch("L1Matched",&matchL1Bit,"matchL1Bit/I");
  l2tree->Branch("seedTower_Et",&seedTowerEt,"seedTower_Et/F");
  l2tree->Branch("Jet_Et",&JetEt,"Jet_Et/F");
 
 
}


L2TauAnalyzer::~L2TauAnalyzer()
{
}



void
L2TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association)
   Handle<LVColl> McInfo; //Handle To The Truth!!!!
   Handle<GenJetCollection> genJets; //Handle To Gen Jets for QCD

   try {
        iEvent.getByLabel(genJets_,genJets);
       }
       catch(...)
       {
	 std::cout <<"No Generated Jet Collection was Found in this Event"<< std::endl;
       }

   try {
        iEvent.getByLabel(mcColl_,McInfo);
       }
       catch(...)
       {
	 std::cout <<"No MCInfoCollection was Found in this Event"<< std::endl;
       }



   try {
     
    
        iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc);//get the handle

       }
       catch(...)
       {
	 std::cout << "No L2TauInfoAssociation  found in the Event" << std::endl;
       }



       //If the Collection exists do work
    if(&(*l2TauInfoAssoc))
     for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p)
     {
       //Retrieve The L2TauIsolationInfo Class from the AssociationMap
       const L2TauIsolationInfo l2info = p->val;
       
        //Retrieve the Jet From the AssociationMap
       const Jet& jet =*(p->key);

 
      
       //match your  jet
       
       MatchElement mcMatch;
       
       if(IsSignal_)
	 mcMatch=match(jet,*McInfo);
       else
	 mcMatch=matchQCD(jet,*genJets);


       //match your jet to L1
       edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
       
       matchL1Bit=0;
       if(iEvent.getByLabel(l1taus_,l1TriggeredTaus))
	 {
    	   std::vector<l1extra::L1JetParticleRef> tauCandRefVec;
	   l1TriggeredTaus->getObjects(trigger::TriggerL1TauJet,tauCandRefVec);


	   if(matchL1(jet,tauCandRefVec))
	     matchL1Bit=1;
	   else
	     matchL1Bit=0;
	 }


       //Fill variables

       matchBit=mcMatch.matched;
       ecalIsol_Et=l2info.ECALIsolConeCut;
       towerIsol_Et=l2info.TowerIsolConeCut;
       cl_Nclusters=l2info.ECALClusterNClusters;
       cl_etaRMS=l2info.ECALClusterEtaRMS;
       cl_phiRMS=l2info.ECALClusterPhiRMS;
       cl_drRMS=l2info.ECALClusterDRRMS;
       seedTowerEt = l2info.SeedTowerEt;
       MCeta =mcMatch.mcEta;
       MCet=mcMatch.mcEt;
       JetEt = jet.et();


       //Fill Tree
       l2tree->Fill();
	   
     } 
	       
     



}



void 
L2TauAnalyzer::beginJob(const edm::EventSetup&)
{

}


void 
L2TauAnalyzer::endJob() {
  l2file->Write();

}

MatchElement 
L2TauAnalyzer::match(const reco::Jet& jet,const LVColl& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 double delta_min=100.;
 double mceta=0;
 double mcet=0;
 
 double matchingDR;

   matchingDR=0.15;


 if(McInfo.size()>0)
  for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),*it);
	  if(delta<matchingDR)
	    {
	      matched=true;
	      if(delta<delta_min)
		{
		  delta_min=delta;
		  mceta=it->eta();
		  mcet=it->Et();
		}
	    }
   }

  //Create Struct and send it out!
  MatchElement match;
  match.matched=matched;
  match.deltar=delta_min;
  match.mcEta = mceta;
  match.mcEt = mcet;


 return match;
}





bool 
L2TauAnalyzer::matchL1(const reco::Jet& jet,std::vector<l1extra::L1JetParticleRef>& tauCandRefVec)
{

  bool match = false;


	   for( unsigned int iL1Tau=0; iL1Tau <tauCandRefVec.size();iL1Tau++)
	     {  
	      
        	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),tauCandRefVec[iL1Tau]->p4().Vect());
		  printf("L1 Match Dr = %f \n",delta);
		  if(delta<0.5)
		    match=true;
	     }

	   return match;
}



MatchElement 
L2TauAnalyzer::matchQCD(const reco::Jet& jet,const reco::GenJetCollection& genjets )
{

  //Loop On the Collection and see if your jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 double delta_min=100.;
 double mceta=0.;
 double mcet=0.;
 
 double matchingDR;
 
  matchingDR=0.3;

 if(genjets.size()>0)
  for(reco::GenJetCollection::const_iterator it = genjets.begin();it!=genjets.end();++it)
   {
          
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),it->p4().Vect());
	  if(delta<matchingDR)
	    {
	      matched=true;
	      if(delta<delta_min)
		{
		  delta_min=delta;
		  mcet=it->et();
		  mceta=it->eta();
		}
	    }
   }

  //Create Struct and send it out!
  MatchElement match;
  match.matched=matched;
  match.mcEta=mceta;
  match.mcEt=mcet;


 return match;
}




