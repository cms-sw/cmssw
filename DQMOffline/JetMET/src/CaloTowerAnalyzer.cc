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
#include "DQMServices/Core/interface/DQMStore.h"

using namespace reco;
using namespace std;

CaloTowerAnalyzer::CaloTowerAnalyzer(const edm::ParameterSet & iConfig)
{

  caloTowersLabel_     = iConfig.getParameter<edm::InputTag>("CaloTowersLabel");
  debug_               = iConfig.getParameter<bool>("Debug");
  finebinning_         = iConfig.getUntrackedParameter<bool>("FineBinning"); 
  FolderName_            = iConfig.getUntrackedParameter<string>("FolderName");
}


void CaloTowerAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  Nevents = 0;
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();

  if (dbe_) {
 
    //TString dirName = "RecoMETV/MET_CaloTowers/";
    dbe_->setCurrentFolder(FolderName_); 
    
    //--Store number of events used
    me["hCT_Nevents"]          = dbe_->book1D("METTask_CT_Nevents","",1,0,1);  
    //--Data integrated over all events and stored by CaloTower(ieta,iphi) 
    me["hCT_et_ieta_iphi"]          = dbe_->book2D("METTask_CT_et_ieta_iphi","",83,-41,42, 72,1,73);  

    me["hCT_emEt_ieta_iphi"]        = dbe_->book2D("METTask_CT_emEt_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_hadEt_ieta_iphi"]       = dbe_->book2D("METTask_CT_hadEt_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_Et_ieta_iphi"]      = dbe_->book2D("METTask_CT_Et_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_outerEt_ieta_iphi"] = dbe_->book2D("METTask_CT_outerEt_ieta_iphi","",83,-41,42, 72,1,73);  
    //me["hCT_energy_ieta_iphi"]      = dbe_->book2D("METTask_CT_energy_ieta_iphi","",83,-41,42, 72,1,73);  
    //me["hCT_outerEnergy_ieta_iphi"] = dbe_->book2D("METTask_CT_outerEnergy_ieta_iphi","",83,-41,42, 72,1,73);  
    //me["hCT_hadEnergy_ieta_iphi"]   = dbe_->book2D("METTask_CT_hadEnergy_ieta_iphi","",83,-41,42, 72,1,73);  
    //me["hCT_emEnergy_ieta_iphi"]    = dbe_->book2D("METTask_CT_emEnergy_ieta_iphi","",83,-41,42, 72,1,73);  
    me["hCT_Occ_ieta_iphi"]         = dbe_->book2D("METTask_CT_Occ_ieta_iphi","",83,-41,42, 72,1,73);  
    //--Data over eta-rings

    // CaloTower values
    if(finebinning_)
      {
	me["hCT_etvsieta"]          = dbe_->book2D("METTask_CT_etvsieta","", 83,-41,42, 10001,0,1001);  
	me["hCT_Minetvsieta"]       = dbe_->book2D("METTask_CT_Minetvsieta","", 83,-41,42, 10001,0,1001);  
	me["hCT_Maxetvsieta"]       = dbe_->book2D("METTask_CT_Maxetvsieta","", 83,-41,42, 10001,0,1001);  
	me["hCT_emEtvsieta"]        = dbe_->book2D("METTask_CT_emEtvsieta","",83,-41,42, 10001,0,1001);  
	me["hCT_hadEtvsieta"]       = dbe_->book2D("METTask_CT_hadEtvsieta","",83,-41,42, 10001,0,1001);  
	me["hCT_outerEtvsieta"] = dbe_->book2D("METTask_CT_outerEtvsieta","",83,-41,42, 10001,0,1001);  
	//me["hCT_energyvsieta"]      = dbe_->book2D("METTask_CT_energyvsieta","",83,-41,42, 10001,0,1001);  
	//me["hCT_outerEnergyvsieta"] = dbe_->book2D("METTask_CT_outerEnergyvsieta","",83,-41,42, 10001,0,1001);  
	//me["hCT_hadEnergyvsieta"]   = dbe_->book2D("METTask_CT_hadEnergyvsieta","",83,-41,42, 10001,0,1001);  
	//me["hCT_emEnergyvsieta"]    = dbe_->book2D("METTask_CT_emEnergyvsieta","",83,-41,42, 10001,0,1001);  

	// Integrated over phi
	me["hCT_Occvsieta"]         = dbe_->book2D("METTask_CT_Occvsieta","",83,-41,42, 84,0,84);  
	me["hCT_SETvsieta"]         = dbe_->book2D("METTask_CT_SETvsieta","",83,-41,42, 20001,0,2001);  
	me["hCT_METvsieta"]         = dbe_->book2D("METTask_CT_METvsieta","",83,-41,42, 20001,0,2001);  
	me["hCT_METPhivsieta"]      = dbe_->book2D("METTask_CT_METPhivsieta","",83,-41,42, 80,-4,4);  
	me["hCT_MExvsieta"]         = dbe_->book2D("METTask_CT_MExvsieta","",83,-41,42, 10001,-500,501);  
	me["hCT_MEyvsieta"]         = dbe_->book2D("METTask_CT_MEyvsieta","",83,-41,42, 10001,-500,501);  
      }
    else 
      {
	
	me["hCT_etvsieta"]          = dbe_->book2D("METTask_CT_etvsieta","", 83,-41,42, 200,-0.5,999.5);
        me["hCT_Minetvsieta"]       = dbe_->book2D("METTask_CT_Minetvsieta","", 83,-41,42, 200,-0.5,999.5);
        me["hCT_Maxetvsieta"]       = dbe_->book2D("METTask_CT_Maxetvsieta","", 83,-41,42, 200,-0.5,999.5);
        me["hCT_emEtvsieta"]        = dbe_->book2D("METTask_CT_emEtvsieta","",83,-41,42, 200,-0.5,999.5);
        me["hCT_hadEtvsieta"]       = dbe_->book2D("METTask_CT_hadEtvsieta","",83,-41,42, 200,-0.5,999.5);
        me["hCT_outerEtvsieta"] = dbe_->book2D("METTask_CT_outerEtvsieta","",83,-41,42, 80,-0.5,399.5);

        //me["hCT_energyvsieta"]      = dbe_->book2D("METTask_CT_energyvsieta","",83,-41,42, 200,-0.5,999.5);
        //me["hCT_outerEnergyvsieta"] = dbe_->book2D("METTask_CT_outerEnergyvsieta","",83,-41,42, 80,-0.5,399.5);
        //me["hCT_hadEnergyvsieta"]   = dbe_->book2D("METTask_CT_hadEnergyvsieta","",83,-41,42, 200,-0.5,999.5);
        //me["hCT_emEnergyvsieta"]    = dbe_->book2D("METTask_CT_emEnergyvsieta","",83,-41,42, 200,-0.5,999.5);

        // Integrated over phi                                                                                                                                                                                 
        me["hCT_Occvsieta"]         = dbe_->book2D("METTask_CT_Occvsieta","",83,-41,42, 73,-0.5,72.5);
        me["hCT_SETvsieta"]         = dbe_->book2D("METTask_CT_SETvsieta","",83,-41,42, 200,-0.5,1999.5);
        me["hCT_METvsieta"]         = dbe_->book2D("METTask_CT_METvsieta","",83,-41,42, 200,-0.5,1999.5);
        me["hCT_METPhivsieta"]      = dbe_->book2D("METTask_CT_METPhivsieta","",83,-41,42, 80,-4,4);
        me["hCT_MExvsieta"]         = dbe_->book2D("METTask_CT_MExvsieta","",83,-41,42, 100,-499.5,499.5);
        me["hCT_MEyvsieta"]         = dbe_->book2D("METTask_CT_MEyvsieta","",83,-41,42, 100,-499.5,499.5);
	
      }
  }
}

void CaloTowerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //----------GREG & CHRIS' idea---///
   float ETTowerMin = -1; //GeV
   float METRingMin = -2; // GeV

  Nevents++;
  me["hCT_Nevents"]->Fill(0);

  // ==========================================================
  // Retrieve!
  // ==========================================================

  edm::Handle<edm::View<Candidate> > towers;
  iEvent.getByLabel(caloTowersLabel_, towers);
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

  //rcr for (calotower = towerCollection->begin(); calotower != towerCollection->end(); calotower++) {
    
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
	  me["hCT_Occ_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi);
	  me["hCT_et_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_ET);
	  me["hCT_emEt_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_EMEt);
	  me["hCT_hadEt_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_HadEt);
	  me["hCT_outerEt_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_OuterEt);

	  //me["hCT_energy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_Energy);
	  //me["hCT_outerEnergy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_OuterEnergy);
	  //me["hCT_hadEnergy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_HadEnergy);
	  //me["hCT_emEnergy_ieta_iphi"]->Fill(Tower_ieta,Tower_iphi,Tower_EMEnergy);
	  
	  me["hCT_etvsieta"]->Fill(Tower_ieta, Tower_ET);
	  me["hCT_emEtvsieta"]->Fill(Tower_ieta, Tower_EMEt);
	  me["hCT_hadEtvsieta"]->Fill(Tower_ieta,Tower_HadEt);
	  me["hCT_outerEtvsieta"]->Fill(Tower_ieta,Tower_OuterEt);
	  //me["hCT_energyvsieta"]->Fill(Tower_ieta,Tower_Energy);
	  //me["hCT_outerEnergyvsieta"]->Fill(Tower_ieta,Tower_OuterEnergy);
	  //me["hCT_hadEnergyvsieta"]->Fill(Tower_ieta ,Tower_HadEnergy);
	  //me["hCT_emEnergyvsieta"]->Fill(Tower_ieta,Tower_EMEnergy);

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
  for (int iEtaRing=0; iEtaRing<83; iEtaRing++)
    { 
      me["hCT_Minetvsieta"]->Fill(iEtaRing-41, MinEt_EtaRing[iEtaRing]);  
      me["hCT_Maxetvsieta"]->Fill(iEtaRing-41, MaxEt_EtaRing[iEtaRing]);  
      
      if (ActiveRing[iEtaRing])
	{
	  if (vMET_EtaRing[iEtaRing].Pt()>METRingMin)
	    {
	      me["hCT_METPhivsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Phi());
	      me["hCT_MExvsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Px());
	      me["hCT_MEyvsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Py());
	      me["hCT_METvsieta"]->Fill(iEtaRing-41, vMET_EtaRing[iEtaRing].Pt());
	    }
	  me["hCT_SETvsieta"]->Fill(iEtaRing-41, SET_EtaRing[iEtaRing]);
	  me["hCT_Occvsieta"]->Fill(iEtaRing-41, NActiveTowers[iEtaRing]);
	}
    }
  
  edm::LogInfo("OutputInfo") << "CT ieta range: " << CTmin_ieta << " " << CTmax_ieta;
  edm::LogInfo("OutputInfo") << "CT iphi range: " << CTmin_iphi << " " << CTmax_iphi;
  
}

void CaloTowerAnalyzer::endJob()
{
} 
