#include "HLTriggerOffline/Tau/interface/L2TauAnalyzer.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

L2TauAnalyzer::L2TauAnalyzer(const edm::ParameterSet& iConfig):
 l2TauInfoAssoc_(iConfig.getParameter<edm::InputTag>("L2InfoAssociationInput")),
 l1Taus_(iConfig.getParameter<edm::InputTag>("L1TauCollection")),
 l1Jets_(iConfig.getParameter<edm::InputTag>("L1JetCollection")),
 rootFile_(iConfig.getParameter<std::string>("outputFileName")),
 IsSignal_(iConfig.getParameter<bool>("IsSignal")),
 mcColl_(iConfig.getParameter<edm::InputTag>("MatchedCollection"))
{
  //File Setup
  l2file = new TFile(rootFile_.c_str(),"recreate");
  //Tree Setup
  l2tree = new TTree("l2tree","Level 2 Tau Tree");


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
  JetEt=0.;
  JetEta=0.;
  L1et=0.;
  L1eta=0.;
  jetEMF = 0.;

  //Setup Branches
  l2tree->Branch("ecalIsolEt",&ecalIsol_Et,"ecalIsolEt/F");
  l2tree->Branch("jetEMF",&jetEMF,"jetEMF/F");
  l2tree->Branch("towerIsolEt",&towerIsol_Et,"towerIsolEt/F");
  l2tree->Branch("clEtaRMS",&cl_etaRMS,"clEtaRMS/F");
  l2tree->Branch("clPhiRMS",&cl_phiRMS,"clPhiRMS/F");
  l2tree->Branch("clDrRMS",&cl_drRMS,"clDrRMS/F");
  l2tree->Branch("mcEta",&MCeta,"mcEta/F");
  l2tree->Branch("mcEt",&MCet,"mcEt/F");
  l2tree->Branch("clNclusters",&cl_Nclusters,"clNclusters/I");
  l2tree->Branch("seedTowerEt",&seedTowerEt,"seedTowerEt/F");
  l2tree->Branch("jetEt",&JetEt,"jetEt/F");
  l2tree->Branch("jetEta",&JetEta,"jetEta/F");
  l2tree->Branch("L1Et",&L1et,"L1Et/F");
  l2tree->Branch("L1Eta",&L1eta,"L1Eta/F");
 
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
   Handle<l1extra::L1JetParticleCollection> L1Taus; //Handle To The L1 Taus
   Handle<l1extra::L1JetParticleCollection> L1Jets; //Handle To The L1 jets

        if(iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc))//get the handle
	  {
	    if(l2TauInfoAssoc->size()>0)
	      for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p)
		{
		  const L2TauIsolationInfo l2info = p->val;
      		  const CaloJet& jet =*(p->key);
       
		  MatchElementL2 mcMatch;
		  mcMatch.matched=false;
		  mcMatch.mcEt=0;
		  mcMatch.mcEta=0;
		  mcMatch.deltar=0;

		  if(IsSignal_) //Get Collection and match it
		    {
	              if(iEvent.getByLabel(mcColl_,McInfo))
			  mcMatch=match(jet,*McInfo);
		    }

		  if((mcMatch.matched&&IsSignal_)||(!IsSignal_))
		    {
		      //Fill variables
		      jetEMF = jet.emEnergyFraction();
		      ecalIsol_Et=l2info.ecalIsolEt();
		      towerIsol_Et=l2info.hcalIsolEt();
		      cl_Nclusters=l2info.nEcalHits();
		      cl_etaRMS=l2info.ecalClusterShape()[0];
		      cl_phiRMS=l2info.ecalClusterShape()[1];
		      cl_drRMS=l2info.ecalClusterShape()[2];
		      seedTowerEt = l2info.seedHcalHitEt();
		      MCeta =mcMatch.mcEta;
		      MCet=mcMatch.mcEt;
		      JetEt = jet.et();
		      JetEta = jet.eta();

		      //Match with L1 and fill
		      L1et=0;
		      L1eta=0;
		      if(iEvent.getByLabel(l1Taus_,L1Taus))
			{
			    MatchElementL2 l1Match;
			    l1Match.matched=false;
			    l1Match.mcEt=0;
			    l1Match.mcEta=0;
			    l1Match.deltar=0;
			    l1Match=match(jet,*L1Taus);
			    if(l1Match.matched)
			      {
				L1et=l1Match.mcEt;
				L1eta=l1Match.mcEta;
			      }
			    //If not matched look at the jet collection
			    else
			      {
				if(iEvent.getByLabel(l1Jets_,L1Jets))
				  {
				    l1Match=match(jet,*L1Taus);
				    if(l1Match.matched)
				      {
					L1et=l1Match.mcEt;
					L1eta=l1Match.mcEta;
				      }

				  }
			      }

			}
		      //Fill Tree
		      l2tree->Fill();
		    }
	   
		}
	  } 
}



void 
L2TauAnalyzer::beginJob()
{

}


void 
L2TauAnalyzer::endJob() {
  l2file->Write();

}

MatchElementL2 
L2TauAnalyzer::match(const reco::Jet& jet,const LVColl& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 double delta_min=100.;
 double mceta=0;
 double mcet=0;
 
 double matchingDR=0.3;




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
  MatchElementL2 match;
  match.matched=matched;
  match.deltar=delta_min;
  match.mcEta = mceta;
  match.mcEt = mcet;


 return match;
}

MatchElementL2 
L2TauAnalyzer::match(const reco::Jet& jet,const l1extra::L1JetParticleCollection& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 double delta_min=100.;
 double mceta=0;
 double mcet=0;
 
 double matchingDR=0.5;




 if(McInfo.size()>0)
  for(l1extra::L1JetParticleCollection::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),it->p4().Vect());
	  if(delta<matchingDR)
	    {
	      matched=true;
	      if(delta<delta_min)
		{
		  delta_min=delta;
		  mceta=it->eta();
		  mcet=it->et();
		}
	    }
   }

  //Create Struct and send it out!
  MatchElementL2 match;
  match.matched=matched;
  match.deltar=delta_min;
  match.mcEta = mceta;
  match.mcEt = mcet;


 return match;
}





