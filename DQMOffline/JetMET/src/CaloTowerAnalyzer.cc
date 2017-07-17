#include "DQMOffline/JetMET/interface/CaloTowerAnalyzer.h"
// author: Bobby Scurlock, University of Florida
// first version 12/18/2006
// modified: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

//#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <TLorentzVector.h>

using namespace reco;
using namespace edm;
using namespace std;

CaloTowerAnalyzer::CaloTowerAnalyzer(const edm::ParameterSet & iConfig)
{

  caloTowersLabel_            = consumes<edm::View<reco::Candidate> > (iConfig.getParameter<edm::InputTag>("CaloTowersLabel"));
  HLTResultsLabel_            = consumes<edm::TriggerResults> (iConfig.getParameter<edm::InputTag>("HLTResultsLabel"));
  HBHENoiseFilterResultLabel_ = consumes<bool> (iConfig.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel"));

  if(iConfig.exists("HLTBitLabels"))
    HLTBitLabel_         = iConfig.getParameter<std::vector<edm::InputTag> >("HLTBitLabels");
  
  debug_               = iConfig.getParameter<bool>("Debug");
  finebinning_         = iConfig.getUntrackedParameter<bool>("FineBinning"); 
  allhist_             = iConfig.getUntrackedParameter<bool>("AllHist"); 
  FolderName_          = iConfig.getUntrackedParameter<std::string>("FolderName");

  hltselection_        = iConfig.getUntrackedParameter<bool>("HLTSelection"); 
  
}


void CaloTowerAnalyzer::dqmbeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  Nevents = 0;
}
    
void CaloTowerAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					edm::Run const & iRun,
					edm::EventSetup const & )
  {
    ibooker.setCurrentFolder(FolderName_); 

    //Store number of events which pass each HLT bit 
    for(unsigned int i = 0 ; i < HLTBitLabel_.size() ; i++ )
      {
	if( HLTBitLabel_[i].label().size() )
	  {
	    hCT_NEvents_HLT.push_back( ibooker.book1D("METTask_CT_"+HLTBitLabel_[i].label(),HLTBitLabel_[i].label(),2,-0.5,1.5) );
	  }
      } 
    
    //--Store number of events used
    hCT_Nevents          = ibooker.book1D("METTask_CT_Nevents","",1,0,1);  
    //--Data integrated over all events and stored by CaloTower(ieta,iphi) 
    hCT_et_ieta_iphi          = ibooker.book2D("METTask_CT_et_ieta_iphi","",83,-41,42, 72,1,73);  
    hCT_et_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_et_ieta_iphi->setAxisTitle("ieta",1);
    hCT_et_ieta_iphi->setAxisTitle("ephi",2);

    hCT_emEt_ieta_iphi        = ibooker.book2D("METTask_CT_emEt_ieta_iphi","",83,-41,42, 72,1,73);  
    hCT_emEt_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_emEt_ieta_iphi->setAxisTitle("ieta",1);
    hCT_emEt_ieta_iphi->setAxisTitle("ephi",2);
    hCT_hadEt_ieta_iphi       = ibooker.book2D("METTask_CT_hadEt_ieta_iphi","",83,-41,42, 72,1,73);  
    hCT_hadEt_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_hadEt_ieta_iphi->setAxisTitle("ieta",1);
    hCT_hadEt_ieta_iphi->setAxisTitle("ephi",2);
    hCT_outerEt_ieta_iphi = ibooker.book2D("METTask_CT_outerEt_ieta_iphi","",83,-41,42, 72,1,73);  
    hCT_outerEt_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_outerEt_ieta_iphi->setAxisTitle("ieta",1);
    hCT_outerEt_ieta_iphi->setAxisTitle("ephi",2);
    hCT_Occ_ieta_iphi         = ibooker.book2D("METTask_CT_Occ_ieta_iphi","",83,-41,42, 72,1,73);  
    hCT_Occ_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_Occ_ieta_iphi->setAxisTitle("ieta",1);
    hCT_Occ_ieta_iphi->setAxisTitle("ephi",2);

    hCT_Occ_EM_Et_ieta_iphi         = ibooker.book2D("METTask_CT_Occ_EM_Et_ieta_iphi","",83,-41,42, 72,1,73);
    hCT_Occ_EM_Et_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_Occ_EM_Et_ieta_iphi->setAxisTitle("ieta",1);
    hCT_Occ_EM_Et_ieta_iphi->setAxisTitle("ephi",2);

    hCT_Occ_HAD_Et_ieta_iphi         = ibooker.book2D("METTask_CT_Occ_HAD_Et_ieta_iphi","",83,-41,42, 72,1,73);
    hCT_Occ_HAD_Et_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_Occ_HAD_Et_ieta_iphi->setAxisTitle("ieta",1);
    hCT_Occ_HAD_Et_ieta_iphi->setAxisTitle("ephi",2);

    hCT_Occ_Outer_Et_ieta_iphi         = ibooker.book2D("METTask_CT_Occ_Outer_Et_ieta_iphi","",83,-41,42, 72,1,73);
    hCT_Occ_Outer_Et_ieta_iphi->getTH2F()->SetOption("colz");
    hCT_Occ_Outer_Et_ieta_iphi->setAxisTitle("ieta",1);
    hCT_Occ_Outer_Et_ieta_iphi->setAxisTitle("ephi",2);

    //--Data over eta-rings

    // CaloTower values
    if(allhist_){
    if(finebinning_)
      {

	hCT_etvsieta          = ibooker.book2D("METTask_CT_etvsieta","", 83,-41,42, 10001,0,1001);  
	hCT_Minetvsieta       = ibooker.book2D("METTask_CT_Minetvsieta","", 83,-41,42, 10001,0,1001);  
	hCT_Maxetvsieta       = ibooker.book2D("METTask_CT_Maxetvsieta","", 83,-41,42, 10001,0,1001);  
	hCT_emEtvsieta        = ibooker.book2D("METTask_CT_emEtvsieta","",83,-41,42, 10001,0,1001);  
	hCT_hadEtvsieta       = ibooker.book2D("METTask_CT_hadEtvsieta","",83,-41,42, 10001,0,1001);  
	hCT_outerEtvsieta = ibooker.book2D("METTask_CT_outerEtvsieta","",83,-41,42, 10001,0,1001);  
	// Integrated over phi

	hCT_Occvsieta         = ibooker.book2D("METTask_CT_Occvsieta","",83,-41,42, 84,0,84);  
	hCT_SETvsieta         = ibooker.book2D("METTask_CT_SETvsieta","",83,-41,42, 20001,0,2001);  
	hCT_METvsieta         = ibooker.book2D("METTask_CT_METvsieta","",83,-41,42, 20001,0,2001);  
	hCT_METPhivsieta      = ibooker.book2D("METTask_CT_METPhivsieta","",83,-41,42, 80,-4,4);  
	hCT_MExvsieta         = ibooker.book2D("METTask_CT_MExvsieta","",83,-41,42, 10001,-500,501);  
	hCT_MEyvsieta         = ibooker.book2D("METTask_CT_MEyvsieta","",83,-41,42, 10001,-500,501);  
      }
    else 
      {
	
	if(allhist_){
	hCT_etvsieta          = ibooker.book2D("METTask_CT_etvsieta","", 83,-41,42, 200,-0.5,999.5);
        hCT_Minetvsieta       = ibooker.book2D("METTask_CT_Minetvsieta","", 83,-41,42, 200,-0.5,999.5);
        hCT_Maxetvsieta       = ibooker.book2D("METTask_CT_Maxetvsieta","", 83,-41,42, 200,-0.5,999.5);
        hCT_emEtvsieta        = ibooker.book2D("METTask_CT_emEtvsieta","",83,-41,42, 200,-0.5,999.5);
        hCT_hadEtvsieta       = ibooker.book2D("METTask_CT_hadEtvsieta","",83,-41,42, 200,-0.5,999.5);
        hCT_outerEtvsieta = ibooker.book2D("METTask_CT_outerEtvsieta","",83,-41,42, 80,-0.5,399.5);
        // Integrated over phi
	}

        hCT_Occvsieta         = ibooker.book2D("METTask_CT_Occvsieta","",83,-41,42, 73,-0.5,72.5);
        hCT_SETvsieta         = ibooker.book2D("METTask_CT_SETvsieta","",83,-41,42, 200,-0.5,1999.5);
        hCT_METvsieta         = ibooker.book2D("METTask_CT_METvsieta","",83,-41,42, 200,-0.5,1999.5);
        hCT_METPhivsieta      = ibooker.book2D("METTask_CT_METPhivsieta","",83,-41,42, 80,-4,4);
        hCT_MExvsieta         = ibooker.book2D("METTask_CT_MExvsieta","",83,-41,42, 100,-499.5,499.5);
        hCT_MEyvsieta         = ibooker.book2D("METTask_CT_MEyvsieta","",83,-41,42, 100,-499.5,499.5);
	
      }
    } // allhist
  }

void CaloTowerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  // Get HLT Results
  edm::Handle<edm::TriggerResults> TheHLTResults;
  iEvent.getByToken( HLTResultsLabel_ , TheHLTResults);

 // **** Get the TriggerResults container
  //triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(theTriggerResultsLabel));
  //edm::Handle<edm::TriggerResults> triggerResults;
  //iEvent.getByToken(triggerResultsToken_, triggerResults);


  bool EventPasses = true;
  // Make sure handle is valid
  if( TheHLTResults.isValid() && hltselection_ )
    {
 
      //Get HLT Names
      const edm::TriggerNames & TheTriggerNames = iEvent.triggerNames(*TheHLTResults);
      
      for( unsigned int index = 0 ; index < HLTBitLabel_.size(); index++)
	{
	  if( HLTBitLabel_[index].label().size() )
	    {
	      //Change the default value since HLT requirement has been issued by the user
	      if( index == 0 ) EventPasses = false; 
	      //Get the HLT bit and check to make sure it is valid
	      unsigned int bit = TheTriggerNames.triggerIndex( HLTBitLabel_[index].label().c_str());
	      if( bit < TheHLTResults->size() )
		{
		  //If any of the HLT names given by the user accept, then the event passes
		  if( TheHLTResults->accept( bit ) && !TheHLTResults->error( bit ) )
		    {
		      EventPasses = true;
		      hCT_NEvents_HLT[index]->Fill(1);
		    }  
		  else 
		    hCT_NEvents_HLT[index]->Fill(0);
		}
	      else
		{
		  edm::LogInfo("OutputInfo") 
		    << "The HLT Trigger Name : " << HLTBitLabel_[index].label() << " is not valid for this trigger table " << std::endl;
		}
	    }
	}
    }

  if( !EventPasses && hltselection_ ) 
    return;
  
  //----------GREG & CHRIS' idea---///
  float ETTowerMin = -1; //GeV
  float METRingMin = -2; // GeV
  
  Nevents++;
  hCT_Nevents->Fill(0);

  // ==========================================================
  // Retrieve!
  // ==========================================================

  edm::Handle<edm::View<Candidate> > towers;
  iEvent.getByToken(caloTowersLabel_, towers);

  if( (!towers.isValid())) {
    //DD:fix print label
    //edm::LogInfo("")<<"CaloTowers "<< caloTowersLabel_<<" not found!"<<std::endl;
    return;
  }

  //HBHENoiseFilterResultToken_=consumes<HBHENoiseFilter>(edm::InputTag(HBHENoiseFilterResultLabel_));
  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByToken(HBHENoiseFilterResultLabel_, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "CaloTowerAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }


  bool bHcalNoiseFilter = HBHENoiseFilterResult;

  if(!bHcalNoiseFilter) return;

  edm::View<Candidate>::const_iterator towerCand = towers->begin();
  
  // ==========================================================
  // Fill Histograms!
  // ==========================================================

  int CTmin_iphi = 99, CTmax_iphi = -99;
  int CTmin_ieta = 99, CTmax_ieta = -99;

  TLorentzVector vMET_EtaRing[83];
  int ActiveRing[83];
  int NActiveTowers[83];
  double SET_EtaRing[83];
  double MinEt_EtaRing[83];
  double MaxEt_EtaRing[83];
  for (int i=0;i<83; i++) 
    {
      ActiveRing[i] = 0;
      NActiveTowers[i] = 0;
      SET_EtaRing[i] = 0;
      MinEt_EtaRing[i] = 0;
      MaxEt_EtaRing[i] = 0;
    }

  //  for (CaloTowerCollection::const_iterator calotower = towers->begin(); calotower != towers->end(); calotower++) 
  for ( ; towerCand != towers->end(); towerCand++)
    {
      const Candidate* candidate = &(*towerCand);
      if (candidate) 
	{
	  const CaloTower* calotower = dynamic_cast<const CaloTower*> (candidate);
	  if (calotower){
	    //math::RhoEtaPhiVector Momentum = calotower->momentum();
	    double Tower_ET = calotower->et();
	  //double Tower_Energy  = calotower->energy();
	  //	  double Tower_Eta = calotower->eta();
	  double Tower_Phi = calotower->phi();
	  //double Tower_EMEnergy = calotower->emEnergy();
	  //double Tower_HadEnergy = calotower->hadEnergy();
	  double Tower_OuterEt = calotower->outerEt();
	  double Tower_EMEt = calotower->emEt();
	  double Tower_HadEt = calotower->hadEt();
	  //int Tower_EMLV1 = calotower->emLvl1();
	  //int Tower_HadLV1 = calotower->hadLv11();
	  int Tower_ieta = calotower->id().ieta();
	  int Tower_iphi = calotower->id().iphi();
	  int EtaRing = 41+Tower_ieta;
	  ActiveRing[EtaRing] = 1;
	  NActiveTowers[EtaRing]++;
	  SET_EtaRing[EtaRing]+=Tower_ET;
	  TLorentzVector v_;
	  v_.SetPtEtaPhiE(Tower_ET, 0, Tower_Phi, Tower_ET);
	  if (Tower_ET>ETTowerMin)
	    vMET_EtaRing[EtaRing]-=v_;
	  
	  // Fill Histograms                                                                                                                                                                                                   
          hCT_Occ_ieta_iphi->Fill(Tower_ieta,Tower_iphi);
          if (calotower->emEt() > 0 && calotower->emEt() + calotower->hadEt() > 0.3)
            hCT_Occ_EM_Et_ieta_iphi->Fill(Tower_ieta,Tower_iphi);
          if (calotower->hadEt() > 0 && calotower->emEt() + calotower->hadEt() > 0.3)
            hCT_Occ_HAD_Et_ieta_iphi->Fill(Tower_ieta,Tower_iphi);
          if (calotower->outerEt() > 0 && calotower->emEt() + calotower->hadEt() > 0.3)
            hCT_Occ_Outer_Et_ieta_iphi->Fill(Tower_ieta,Tower_iphi);

	  hCT_et_ieta_iphi->Fill(Tower_ieta,Tower_iphi,Tower_ET);
	  hCT_emEt_ieta_iphi->Fill(Tower_ieta,Tower_iphi,Tower_EMEt);
	  hCT_hadEt_ieta_iphi->Fill(Tower_ieta,Tower_iphi,Tower_HadEt);
	  hCT_outerEt_ieta_iphi->Fill(Tower_ieta,Tower_iphi,Tower_OuterEt);

	  if (allhist_){
	  hCT_etvsieta->Fill(Tower_ieta, Tower_ET);
	  hCT_emEtvsieta->Fill(Tower_ieta, Tower_EMEt);
	  hCT_hadEtvsieta->Fill(Tower_ieta,Tower_HadEt);
	  hCT_outerEtvsieta->Fill(Tower_ieta,Tower_OuterEt);
	  }

	  if (Tower_ET > MaxEt_EtaRing[EtaRing])
	    MaxEt_EtaRing[EtaRing] = Tower_ET;
	  if (Tower_ET < MinEt_EtaRing[EtaRing] && Tower_ET>0)
	    MinEt_EtaRing[EtaRing] = Tower_ET;
	  
	  
	  if (Tower_ieta < CTmin_ieta) CTmin_ieta = Tower_ieta;
	  if (Tower_ieta > CTmax_ieta) CTmax_ieta = Tower_ieta;
	  if (Tower_iphi < CTmin_iphi) CTmin_iphi = Tower_iphi;
	  if (Tower_iphi > CTmax_iphi) CTmax_iphi = Tower_iphi;
	  } //end if (calotower) ..
	} // end if(candidate) ...
    
    } // end loop over towers
  
      // Fill eta-ring MET quantities
  if (allhist_){
  for (int iEtaRing=0; iEtaRing<83; iEtaRing++)
    { 
      hCT_Minetvsieta->Fill(iEtaRing-41, MinEt_EtaRing[iEtaRing]);  
      hCT_Maxetvsieta->Fill(iEtaRing-41, MaxEt_EtaRing[iEtaRing]);  
      
      if (ActiveRing[iEtaRing])
	{
	  if (vMET_EtaRing[iEtaRing].Pt()>METRingMin)
	    {
	      hCT_METPhivsieta->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Phi());
	      hCT_MExvsieta->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Px());
	      hCT_MEyvsieta->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Py());
	      hCT_METvsieta->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Pt());
	    }
	  hCT_SETvsieta->Fill(iEtaRing-41, SET_EtaRing[iEtaRing]);
	  hCT_Occvsieta->Fill(iEtaRing-41, NActiveTowers[iEtaRing]);
	}
    } // ietaring
  }   // allhist_
 
  edm::LogInfo("OutputInfo") << "CT ieta range: " << CTmin_ieta << " " << CTmax_ieta;
  edm::LogInfo("OutputInfo") << "CT iphi range: " << CTmin_iphi << " " << CTmax_iphi;
  
}

