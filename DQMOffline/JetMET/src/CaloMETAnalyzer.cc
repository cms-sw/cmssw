/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/08/26 19:17:30 $
 *  $Revision: 1.2 $
 *  \author F. Chlebana - Fermilab
 */

#include "DQMOffline/JetMET/src/CaloMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h" // Added temporarily by KH

#include <string>
using namespace std;
using namespace edm;

CaloMETAnalyzer::CaloMETAnalyzer(const edm::ParameterSet& pSet) {

  cout<<"[CaloMETAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}


CaloMETAnalyzer::~CaloMETAnalyzer() { }


void CaloMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "caloMETAnalyzer";

  LogTrace(metname)<<"[CaloMETAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/CaloMETAnalyzer");

  jetME = dbe->book1D("caloMETReco", "caloMETReco", 3, 1, 4);
  jetME->setBinLabel(1,"CaloMET",1);

  hNevents                = dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  hCaloMEx                = dbe->book1D("METTask_CaloMEx",   "METTask_CaloMEx"   ,500,-500,500);
  hCaloMEy                = dbe->book1D("METTask_CaloMEy",   "METTask_CaloMEy"   ,500,-500,500);
  hCaloEz                 = dbe->book1D("METTask_CaloEz",    "METTask_CaloEz"    ,500,-500,500);
  hCaloMETSig             = dbe->book1D("METTask_CaloMETSig","METTask_CaloMETSig",51,0,51);
  hCaloMET                = dbe->book1D("METTask_CaloMET",   "METTask_CaloMET"   ,500,0,1000);
  hCaloMETPhi             = dbe->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-4,4);
  hCaloSumET              = dbe->book1D("METTask_CaloSumET", "METTask_CaloSumET" ,1000,0,2000);
  hCaloMExLS              = dbe->book2D("METTask_CaloMEx_LS","METTask_CaloMEx_LS",200,-200,200,500,0.,500.);
  hCaloMEyLS              = dbe->book2D("METTask_CaloMEy_LS","METTask_CaloMEy_LS",200,-200,200,500,0.,500.);

  hCaloMaxEtInEmTowers    = dbe->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,1000,0,2000);
  hCaloMaxEtInHadTowers   = dbe->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,1000,0,2000);
  hCaloEtFractionHadronic = dbe->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
  hCaloEmEtFraction       = dbe->book1D("METTask_CaloEmEtFraction",      "METTask_CaloEmEtFraction"      ,100,0,1);

  hCaloHadEtInHB          = dbe->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",1000,0,2000);
  hCaloHadEtInHO          = dbe->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",1000,0,2000);
  hCaloHadEtInHE          = dbe->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",1000,0,2000);
  hCaloHadEtInHF          = dbe->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",1000,0,2000);
//hCaloHadEtInEB          = dbe->book1D("METTask_CaloHadEtInEB","METTask_CaloHadEtInEB",1000,0,2000);
//hCaloHadEtInEE          = dbe->book1D("METTask_CaloHadEtInEE","METTask_CaloHadEtInEE",1000,0,2000);
  hCaloEmEtInHF           = dbe->book1D("METTask_CaloEmEtInHF" ,"METTask_CaloEmEtInHF" ,1000,0,2000);
  hCaloEmEtInEE           = dbe->book1D("METTask_CaloEmEtInEE" ,"METTask_CaloEmEtInEE" ,1000,0,2000);
  hCaloEmEtInEB           = dbe->book1D("METTask_CaloEmEtInEB" ,"METTask_CaloEmEtInEB" ,1000,0,2000);

  hCaloMExNoHF            = dbe->book1D("METTask_CaloMExNoHF",   "METTask_CaloMExNoHF"   ,500,-500,500);
  hCaloMEyNoHF            = dbe->book1D("METTask_CaloMEyNoHF",   "METTask_CaloMEyNoHF"   ,500,-500,500);
  hCaloEzNoHF             = dbe->book1D("METTask_CaloEzNoHF",    "METTask_CaloEzNoHF"    ,500,-500,500);
  hCaloMETSigNoHF         = dbe->book1D("METTask_CaloMETSigNoHF","METTask_CaloMETSigNoHF",51,0,51);
  hCaloMETNoHF            = dbe->book1D("METTask_CaloMETNoHF",   "METTask_CaloMETNoHF"   ,1000,0,1000);
  hCaloMETPhiNoHF         = dbe->book1D("METTask_CaloMETPhiNoHF","METTask_CaloMETPhiNoHF",80,-4,4);
  hCaloSumETNoHF          = dbe->book1D("METTask_CaloSumETNoHF", "METTask_CaloSumETNoHF" ,1000,0,2000);
  hCaloMExNoHFLS          = dbe->book2D("METTask_CaloMExNoHF_LS","METTask_CaloMExNoHF_LS",200,-200,200,500,0.,500.);
  hCaloMEyNoHFLS          = dbe->book2D("METTask_CaloMEyNoHF_LS","METTask_CaloMEyNoHF_LS",200,-200,200,500,0.,500.);

}

void CaloMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const reco::CaloMET& calomet, const reco::CaloMET& calometNoHF) {

  LogTrace(metname)<<"[CaloMETAnalyzer] Analyze CaloMET";

  jetME->Fill(1);

  // ==========================================================
  // Temporarily added by KH to get metNoHF on-the-fly
  //
  // get collection of towers
  edm::Handle<CaloTowerCollection> calotowers;
  iEvent.getByLabel("towerMaker", calotowers);   
  const CaloTowerCollection* inputCol = calotowers.product();
  CaloTowerCollection::const_iterator calotower;

  double sumet=0.;
  double metx=0.;
  double mety=0.;

  double sumetNoHF=0.;
  double metxNoHF=0.;
  double metyNoHF=0.;

  double sum_emetEB=0.;
  double sum_hadetHB=0.;
  double sum_emetEE=0.;
  double sum_hadetHE=0.;
  double sum_emetHF=0.;
  double sum_hadetHF=0.;
  double sum_hadetHO=0.;

  double metx_emetEB=0.;
  double metx_hadetHB=0.;
  double metx_emetEE=0.;
  double metx_hadetHE=0.;
  double metx_emetHF=0.;
  double metx_hadetHF=0.;
  double metx_hadetHO=0.;

  double mety_emetEB=0.;
  double mety_hadetHB=0.;
  double mety_emetEE=0.;
  double mety_hadetHE=0.;
  double mety_emetHF=0.;
  double mety_hadetHF=0.;
  double mety_hadetHO=0.;

  double etthreshold=0.5;

  for( calotower = inputCol->begin(); calotower != inputCol->end(); ++calotower )
    {
      double phi   = calotower->phi();
      //double eta   = calotower->eta();
      //double e     = calotower->energy();
      double et    = calotower->et();
      //
      // Recompute sumet and met
      if (et>etthreshold){
      sumet += et ;
      metx  += et*cos(phi) ;
      mety  += et*sin(phi) ;       
      // }
      //
      /*
      printf(" towers: phi=%7.3f eta=%7.3f et=%7.3f emet=%7.3f hadet=%7.3f totet=%7.3f outeret=%7.3f\n",
	     phi,eta,et,calotower->emEt(),calotower->hadEt(),
	     calotower->emEt()+calotower->hadEt(),calotower->outerEt());
      */
      //
      // sub-detectors
      bool hadIsDone = false;
      bool emIsDone = false;       
      int cell = calotower->constituentsSize();
      //printf("cell=%3d\n",cell);
      //
      while ( --cell >= 0 && (!hadIsDone || !emIsDone)){
	DetId id = calotower->constituent( cell );       
	//
	//--- HCAL
	if (!hadIsDone && id.det() == DetId::Hcal){
	  if (hadIsDone) printf("***WARNING*** hcal twice\n");
	  //printf("hcal\n");
	  HcalSubdetector subdet = HcalDetId(id).subdet();
	  //--- Which HCAL?
	  if( subdet == HcalBarrel || subdet == HcalOuter ){
	    //printf("hcal barrel or outer\n");	     
	    sum_hadetHB  += calotower->hadEt();
	    sum_hadetHO  += calotower->outerEt();
	    metx_hadetHB -= calotower->hadEt()*cos(phi);
	    metx_hadetHO -= calotower->outerEt()*cos(phi);
	    mety_hadetHB -= calotower->hadEt()*sin(phi);
	    mety_hadetHO -= calotower->outerEt()*sin(phi);
	  }
	  else if( subdet == HcalEndcap ){
	    //printf("hcal endcap\n");	     
	    sum_hadetHE  += calotower->hadEt();
	    metx_hadetHE -= calotower->hadEt()*cos(phi);
	    mety_hadetHE -= calotower->hadEt()*sin(phi);
	  }
	  else if( subdet == HcalForward ){
	    //printf("hcal forward\n");	     
	    sum_hadetHF  += calotower->hadEt();
	    sum_emetHF   += calotower->emEt();
	    metx_hadetHF -= calotower->hadEt()*cos(phi);
	    metx_emetHF  -= calotower->emEt()*cos(phi);
	    mety_hadetHF -= calotower->hadEt()*sin(phi);
	    mety_emetHF  -= calotower->emEt()*sin(phi);
	  }	     
	  hadIsDone = true;
	}
	//
	//--- ECAL
	if (!emIsDone && id.det() == DetId::Ecal){
	  if (emIsDone) printf("***WARNING*** em twice\n");
	  //printf("ecal\n");
	  EcalSubdetector subdet = EcalSubdetector( id.subdetId() );
	  //--- Which ECAL?
	  if( subdet == EcalBarrel ){
	    //printf("ecal barrel\n");	     
	    sum_emetEB  += calotower->emEt();
	    metx_emetEB -= calotower->emEt()*cos(phi);
	    mety_emetEB -= calotower->emEt()*sin(phi);
	  }
	  else if( subdet == EcalEndcap ){
	    //printf("ecal endcap\n");	     
	    sum_emetEE  += calotower->emEt();
	    metx_emetEE -= calotower->emEt()*cos(phi);
	    mety_emetEE -= calotower->emEt()*sin(phi);
	  }
	  emIsDone = true;
	}	 
      }
      //
      } // et-threshold
    }
  //--- Temporary addition ends

  // ==========================================================
  // Reconstructed MET Information
  double caloSumET  = calomet.sumEt();
  double caloMETSig = calomet.mEtSig();
  double caloEz     = calomet.e_longitudinal();
  double caloMET    = calomet.pt();
  double caloMEx    = calomet.px();
  double caloMEy    = calomet.py();
  double caloMETPhi = calomet.phi();

  double caloMaxEtInEMTowers    = calomet.maxEtInEmTowers();
  double caloMaxEtInHadTowers   = calomet.maxEtInHadTowers();
  double caloEtFractionHadronic = calomet.etFractionHadronic();
  double caloEmEtFraction       = calomet.emEtFraction();

  double caloHadEtInHB = calomet.hadEtInHB();
  double caloHadEtInHO = calomet.hadEtInHO();
  double caloHadEtInHE = calomet.hadEtInHE();
  double caloHadEtInHF = calomet.hadEtInHF();
  double caloEmEtInEB  = calomet.emEtInEB();
  double caloEmEtInEE  = calomet.emEtInEE();
  double caloEmEtInHF  = calomet.emEtInHF();

  double caloSumETNoHF  = calometNoHF.sumEt();
  double caloMETSigNoHF = calometNoHF.mEtSig();
  double caloEzNoHF     = calometNoHF.e_longitudinal();
  double caloMETNoHF    = calometNoHF.pt();
  double caloMExNoHF    = calometNoHF.px();
  double caloMEyNoHF    = calometNoHF.py();
  double caloMETPhiNoHF = calometNoHF.phi();

  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  /****
  cout << " Run: "              << iEvent.id().run()
       << " Event: "            << iEvent.id().event()
       << " LumiSection: "      << iEvent.luminosityBlock() 
       << " evtCounter: "       << evtCounter 
       << " myLumiosityBlock: " << myLuminosityBlock 
       << endl;
  ***/
  //

  //--- temporary fix of the threshold problem in metNoHF
  sumetNoHF = sum_hadetHB +sum_hadetHE  +sum_emetEB  +sum_emetEE;
  metxNoHF  = metx_emetEB +metx_hadetHB +metx_emetEE +metx_hadetHE;
  metyNoHF  = mety_emetEB +mety_hadetHB +mety_emetEE +mety_hadetHE;
  math::XYZTLorentzVector METNoHFfix(metxNoHF, metyNoHF, 0, pow(metxNoHF*metxNoHF+metyNoHF*metyNoHF,0.5));
  caloSumETNoHF  = sumetNoHF;
  if (sumet>0.)
  caloMETSigNoHF = METNoHFfix.pt()/sqrt(sumet);
  caloMETNoHF    = METNoHFfix.pt();
  caloMExNoHF    = METNoHFfix.px();
  caloMEyNoHF    = METNoHFfix.py();
  caloMETPhiNoHF = METNoHFfix.phi();
  //caloEzNoHF not fixed
  //--- temporary fix ends

  hCaloMEx->Fill(caloMEx);
  hCaloMEy->Fill(caloMEy);
  hCaloMET->Fill(caloMET);
  hCaloMETPhi->Fill(caloMETPhi);
  hCaloSumET->Fill(caloSumET);
  hCaloMETSig->Fill(caloMETSig);
  hCaloEz->Fill(caloEz);
  hCaloMExLS->Fill(caloMEx,myLuminosityBlock);
  hCaloMEyLS->Fill(caloMEy,myLuminosityBlock);

  hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
  hCaloEmEtFraction->Fill(caloEmEtFraction);

  hCaloHadEtInHB->Fill(caloHadEtInHB);
  hCaloHadEtInHO->Fill(caloHadEtInHO);
  hCaloHadEtInHE->Fill(caloHadEtInHE);
  hCaloHadEtInHF->Fill(caloHadEtInHF);
  hCaloEmEtInEB->Fill(caloEmEtInEB);
  hCaloEmEtInEE->Fill(caloEmEtInEE);
  hCaloEmEtInHF->Fill(caloEmEtInHF);

  hCaloMExNoHF->Fill(caloMExNoHF);
  hCaloMEyNoHF->Fill(caloMEyNoHF);
  hCaloMETNoHF->Fill(caloMETNoHF);
  hCaloMETPhiNoHF->Fill(caloMETPhiNoHF);
  hCaloSumETNoHF->Fill(caloSumETNoHF);
  hCaloMETSigNoHF->Fill(caloMETSigNoHF);
  hCaloEzNoHF->Fill(caloEzNoHF);
  hCaloMExNoHFLS->Fill(caloMExNoHF,myLuminosityBlock);
  hCaloMEyNoHFLS->Fill(caloMEyNoHF,myLuminosityBlock);

}
