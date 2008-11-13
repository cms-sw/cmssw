// -*- C++ -*-
//
// Package:    DQMOffline/JetMET
// Class:      DataCertificationJetMET
// 
/**\class DataCertificationJetMET DataCertificationJetMET.cc DQMOffline/JetMET/src/DataCertificationJetMET.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Frank Chlebana"
//         Created:  Sun Oct  5 13:57:25 CDT 2008
// $Id: DataCertificationJetMET.cc,v 1.17 2008/11/13 07:59:03 hatake Exp $
//
//


// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

// Some switches
#define NJetAlgo 4
#define NL3Flags 3
#define DEBUG    0
#define METFIT   0

//
// class decleration
//

class DataCertificationJetMET : public edm::EDAnalyzer {
   public:
      explicit DataCertificationJetMET(const edm::ParameterSet&);
      ~DataCertificationJetMET();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual int data_certificate_met(double, double, double, double);
      virtual int data_certificate_metfit(double, double, double, double);
      virtual void fitd(TH1F*, TF1*, TF1*, TF1*, int);
      virtual void fitdd(TH1D*, TF1*, TF1*, TF1*, int);

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   DQMStore * dbe;
  //DQMStore * rdbe;
   edm::Service<TFileService> fs_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DataCertificationJetMET::DataCertificationJetMET(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  // now do what ever initialization is needed
}


DataCertificationJetMET::~DataCertificationJetMET()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DataCertificationJetMET::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
   
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
#endif

}

// ------------ method called once each job just before starting event loop  ------------
void 
DataCertificationJetMET::beginJob(const edm::EventSetup&)
{
  
  int verbose  = 0;
  int testType = 1; 

  //----------------------------------------------------------------
  // Open input files
  //----------------------------------------------------------------

  verbose   = conf_.getUntrackedParameter<int>("Verbose");
  testType  = conf_.getUntrackedParameter<int>("TestType");
  //
  // testType 0: no comparison with histograms
  //          1: KS test
  //          2: Chi2 test

  std::string filename    = conf_.getUntrackedParameter<std::string>("fileName");
  if (DEBUG) std::cout << "FileName           = " << filename    << std::endl;

  std::string reffilename;
  if (testType>=1){
    reffilename = conf_.getUntrackedParameter<std::string>("refFileName");
    if (DEBUG) std::cout << "Reference FileName = " << reffilename << std::endl;
  }

  // -- Current & Reference Run
  //---------------------------------------------
  dbe = edm::Service<DQMStore>().operator->();
  dbe->open(filename);
  if (testType>=1) dbe->open(reffilename);

  std::vector<MonitorElement*> mes = dbe->getAllContents("");
  if (DEBUG) std::cout << "found " <<  mes.size() << " monitoring elements!" << std::endl;

  dbe->setCurrentFolder("/");
  std::string currDir = dbe->pwd();
  if (DEBUG) std::cout << "--- Current Directory " << currDir << std::endl;

  std::vector<std::string> subDirVec = dbe->getSubdirs();

  std::string RunDir;
  std::string RunNum;
  int         RunNumber;
  std::string RefRunDir;
  std::string RefRunNum;
  int         RefRunNumber;

  // 
  std::vector<std::string>::const_iterator ic = subDirVec.begin();

  // *** If the same file is read in then we have only one subdirectory
  int ind = 0;
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    if (ind == 0) {
      RefRunDir = *ic;
      RefRunNum = *ic;
      RunDir = *ic;
      RunNum = *ic;
    }
    if (ind == 1) {
      RunDir = *ic;
      RunNum = *ic;
    }
    if (DEBUG) std::cout << "-XXX- Dir = >>" << ic->c_str() << "<<" << std::endl;
    ind++;
  }

  //
  // Current
  //
  if (RunDir == "JetMET") {
    RunDir = "";
    if (DEBUG) std::cout << "-XXX- RunDir = >>" << RunDir.c_str() << "<<" << std::endl;
  }
  RunNum.erase(0,4);
  RunNumber = atoi(RunNum.c_str());
  if (DEBUG) std::cout << "--- >>" << RunNumber << "<<" << std::endl;

  //
  // Reference
  //
  if (testType>=1){

    if (RefRunDir == "JetMET") {
      RefRunDir = "";
      if (DEBUG) std::cout << "-XXX- RefRunDir = >>" << RefRunDir.c_str() << "<<" << std::endl;
    }
    RefRunNum.erase(0,4);
    RefRunNumber = atoi(RefRunNum.c_str());
    if (DEBUG) std::cout << "--- >>" << RefRunNumber << "<<" << std::endl;

  }
  //  ic++;

  //----------------------------------------------------------------
  // Book integers/histograms for data certification results
  //----------------------------------------------------------------

  std::string Jet_Tag_L2[NJetAlgo];
  Jet_Tag_L2[0] = "JetMET_Jet_ICone";
  Jet_Tag_L2[1] = "JetMET_Jet_SISCone";
  Jet_Tag_L2[2] = "JetMET_Jet_PFlow";
  Jet_Tag_L2[3] = "JetMET_Jet_JPT";

  std::string Jet_Tag_L3[NJetAlgo][NL3Flags];
  Jet_Tag_L3[0][0] = "JetMET_Jet_ICone_Barrel";
  Jet_Tag_L3[0][1] = "JetMET_Jet_ICone_EndCap";
  Jet_Tag_L3[0][2] = "JetMET_Jet_ICone_Forward";
  Jet_Tag_L3[1][0] = "JetMET_Jet_SISCone_Barrel";
  Jet_Tag_L3[1][1] = "JetMET_Jet_SISCone_EndCap";
  Jet_Tag_L3[1][2] = "JetMET_Jet_SISCone_Forward";
  Jet_Tag_L3[2][0] = "JetMET_Jet_PFlow_Barrel";
  Jet_Tag_L3[2][1] = "JetMET_Jet_PFlow_EndCap";
  Jet_Tag_L3[2][2] = "JetMET_Jet_PFlow_Forward";
  Jet_Tag_L3[3][0] = "JetMET_Jet_JPT_Barrel";
  Jet_Tag_L3[3][1] = "JetMET_Jet_JPT_EndCap";
  Jet_Tag_L3[3][2] = "JetMET_Jet_JPT_Forward";

  if (DEBUG) std::cout << RunDir << std::endl;
  dbe->setCurrentFolder("/JetMET/EventInfo/Certification/");    

  //
  // Layer 1
  //---------
  MonitorElement* mJetDCFL1 = dbe->bookFloat("JetMET_Jet");
  MonitorElement* mMETDCFL1 = dbe->bookFloat("JetMET_MET");

  //
  // Layer 2
  //---------
  MonitorElement* mJetDCFL2[10];
  int iL2JetTags=0;
  for (int itag=0; itag<=NJetAlgo; itag++){
    mJetDCFL2[iL2JetTags] = dbe->bookFloat(Jet_Tag_L2[itag]);
    iL2JetTags++;
  }

  //MonitorElement* mMETDCFL2[10];
  //int iL2METTags=0;
  //mMETDCFL2[iL2METTags] = dbe->bookFloat("JetMET_MET");
  //iL2METTags++;

  //
  // Layer 3
  //---------
  MonitorElement* mJetDCFL3[20];
  int iL3JetTags=0;
  for (int ialg=0; ialg<NJetAlgo; ialg++){
    for (int idet=0; idet<3; idet++){
      mJetDCFL3[iL3JetTags]= dbe->bookFloat(Jet_Tag_L3[ialg][idet]);
      iL3JetTags++;
    }
  }

  MonitorElement* mMETDCFL3[20];
  int iL3METTags=0;
  mMETDCFL3[iL3METTags]= dbe->bookFloat("JetMET_MET_All");
  iL3METTags++;
  mMETDCFL3[iL3METTags]= dbe->bookFloat("JetMET_MET_NoHF");
  iL3METTags++;

  //----------------------------------------------------------------
  // Data certification starts
  //----------------------------------------------------------------

  //
  // Number of lumi section bins
  const int nLSBins=500;

  //-----------------------------
  // Jet DQM Data Certification
  //-----------------------------
  Double_t test_Pt, test_Eta, test_Phi, test_Constituents, test_HFrac;
  test_Pt = test_Eta = test_Phi = test_Constituents = test_HFrac = 0;
  
  Double_t test_Pt_Barrel,  test_Phi_Barrel;
  Double_t test_Pt_EndCap,  test_Phi_EndCap;
  Double_t test_Pt_Forward, test_Phi_Forward;
  test_Pt_Barrel  = test_Phi_Barrel  = 0;
  test_Pt_EndCap  = test_Phi_EndCap  = 0;
  test_Pt_Forward = test_Phi_Forward = 0;

  Int_t Jet_DCF_L1;
  Int_t Jet_DCF_L2[NJetAlgo];
  Int_t Jet_DCF_L3[NJetAlgo][NL3Flags];
  
  std::string refHistoName;
  std::string newHistoName;

  MonitorElement * meNew;
  MonitorElement * meRef;

  // --- Loop over jet algorithms for Layer 2
  for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    

    // *** Kludge to allow using root files written by stand alone job
    if (iAlgo == 0) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/IterativeConeJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/IterativeConeJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/IterativeConeJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/IterativeConeJets/";
      }
    }
    if (iAlgo == 1) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/SISConeJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/SISConeJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/SISConeJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/SISConeJets/";
      }
    }
    if (iAlgo == 2) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/PFJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/PFJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/PFJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/PFJets/";
      }
    }
    if (iAlgo == 3) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/JPT/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/JPT/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/JPT/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/JPT/";
      }
    }

    // ----------------
    // --- Layer 2

    test_Pt           = 0.;
    test_Eta          = 0.;
    test_Phi          = 0.;
    test_Constituents = 0.;
    test_HFrac        = 0.;

    meRef = dbe->get(refHistoName+"Pt");
    meNew = dbe->get(newHistoName+"Pt");    
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Pt = newHisto->KolmogorovTest(refHisto,"UO");
	  break;
	case 2 :
	  test_Pt = newHisto->Chi2Test(refHisto);
	  break;
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Pt << std::endl;    
      }
    }

    meRef = dbe->get(refHistoName+"Eta");
    meNew = dbe->get(newHistoName+"Eta");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Eta = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_Eta = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Eta << std::endl;    
      }
    }

    meRef = dbe->get(refHistoName+"Phi");
    meNew = dbe->get(newHistoName+"Phi");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Phi = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_Phi = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Phi << std::endl;    
      }
    }
     
    meRef = dbe->get(refHistoName+"Constituents");
    meNew = dbe->get(newHistoName+"Constituents");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :	  
	  test_Constituents = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Constituents = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Constituents << std::endl;    
      }
    }
     
    meRef = dbe->get(refHistoName+"HFrac");
    meNew = dbe->get(newHistoName+"HFrac");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_HFrac = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_HFrac = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_HFrac << std::endl;    	
      }
    }

    if (verbose)
    std::cout << "--- Layer 2 Algo "
              << iAlgo    << " "
              << test_Pt  << " "
              << test_Eta << " "
              << test_Phi << " "
              << test_Constituents << " "
              << test_HFrac << std::endl;

    if ( (test_Pt     > 0.95) && (test_Eta          > 0.95) && 
	 (test_Phi    > 0.95) && (test_Constituents > 0.95) && 
	 (test_HFrac  > 0.95) )  {      
      Jet_DCF_L2[iAlgo] = 1;
    } else {
      Jet_DCF_L2[iAlgo] = 0;
    }
    // --- Fill DC results histogram
    mJetDCFL2[iAlgo]->Fill(double(Jet_DCF_L2[iAlgo]));
      
    // ----------------
    // --- Layer 3
    // --- Barrel

    test_Pt_Barrel   = 0.;
    test_Phi_Barrel  = 0.;
    test_Pt_EndCap   = 0.;
    test_Phi_EndCap  = 0.;
    test_Pt_Forward  = 0.;
    test_Phi_Forward = 0.;


    meRef = dbe->get(refHistoName+"Pt_Barrel");
    meNew = dbe->get(newHistoName+"Pt_Barrel");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Pt_Barrel = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Pt_Barrel = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Pt_Barrel << std::endl;    	
      }
    }

    meRef = dbe->get(refHistoName+"Phi_Barrel");
    meNew = dbe->get(newHistoName+"Phi_Barrel");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Phi_Barrel = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Phi_Barrel = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Phi_Barrel << std::endl;    	
      }
    }

    // --- EndCap
    meRef = dbe->get(refHistoName+"Pt_EndCap");
    meNew = dbe->get(newHistoName+"Pt_EndCap");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Pt_EndCap = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Pt_EndCap = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Pt_EndCap << std::endl;    	
      }
    }

    meRef = dbe->get(refHistoName+"Phi_EndCap");
    meNew = dbe->get(newHistoName+"Phi_EndCap");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Phi_EndCap = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Phi_EndCap = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Phi_EndCap << std::endl;    	

      }
    }

    // --- Forward
    meRef = dbe->get(refHistoName+"Pt_Forward");
    meNew = dbe->get(newHistoName+"Pt_Forward");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Pt_Forward = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_Pt_Forward = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Pt_Forward << std::endl;    	
      }
    }

    meRef = dbe->get(refHistoName+"Phi_Forward");
    meNew = dbe->get(newHistoName+"Phi_Forward");
    if ((meRef) && (meNew)) {
      TH1F *refHisto = meRef->getTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType) {
	case 1 :
	  test_Phi_Forward = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Phi_Forward = newHisto->Chi2Test(refHisto);
	}
	if (verbose > 0) std::cout << ">>> Test (" << testType 
				   << ") Result = " << test_Phi_Forward << std::endl;    	

      }
    }

    if (verbose)
    std::cout << "--- Layer 3 Algo "
              << iAlgo    << " "
              << test_Pt_Barrel    << " "
              << test_Phi_Barrel   << " "
              << test_Pt_EndCap    << " "
              << test_Phi_EndCap   << " "
              << test_Pt_Forward   << " "
              << test_Phi_Forward  << " "
              << std::endl;

    if ( (test_Pt_Barrel > 0.95) && (test_Phi_Barrel > 0.95) ) {
      Jet_DCF_L3[iAlgo][0] = 1;
    } else {
      Jet_DCF_L3[iAlgo][0] = 0;
    }
    mJetDCFL3[iAlgo*NL3Flags+0]->Fill(double(Jet_DCF_L3[iAlgo][0]));

    if ( (test_Pt_EndCap > 0.95) && (test_Phi_EndCap > 0.95) ) {
      Jet_DCF_L3[iAlgo][1] = 1;
    } else {
      Jet_DCF_L3[iAlgo][1] = 0;
    }
    // --- Fill DC results histogram
    mJetDCFL3[iAlgo*NL3Flags+1]->Fill(double(Jet_DCF_L3[iAlgo][1]));

    if ( (test_Pt_Forward > 0.95) && (test_Phi_Forward > 0.95) ) {
      Jet_DCF_L3[iAlgo][2] = 1;
    } else {
      Jet_DCF_L3[iAlgo][2] = 0;
    }
    // --- Fill DC results histogram
    mJetDCFL3[iAlgo*NL3Flags+2]->Fill(double(Jet_DCF_L3[iAlgo][2]));

  }
  // --- End of loop over jet algorithms

  // Layer 1
  //---------
  Jet_DCF_L1= 1;
  for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {   
    if (Jet_DCF_L2[iAlgo] == 0) Jet_DCF_L1 = 0;
  }
  mJetDCFL1->Fill(double(Jet_DCF_L1));

  // JET Data Certification Results
  if (DEBUG) {
    std::cout << std::endl;
    //
    //--- Layer 1
    //
    printf("%6d %15d %-35s %10d\n",RunNumber,0,"JetMET", Jet_DCF_L1);
    //
    //--- Layer 2
    //
    for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    
      printf("%6d %15d %-35s %10d\n",RunNumber,0,Jet_Tag_L2[iAlgo].c_str(), Jet_DCF_L2[iAlgo]);
    }
    //
    //--- Layer 3
    //
    for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    
      for (int iL3Flag=0; iL3Flag<NL3Flags; iL3Flag++) {    
	printf("%6d %15d %-35s %10d\n",RunNumber,0,Jet_Tag_L3[iAlgo][iL3Flag].c_str(), Jet_DCF_L3[iAlgo][iL3Flag]);
      }
    }
    std::cout << std::endl;    
  }

  //  return;


  //-----------------------------
  // MET DQM Data Certification
  //-----------------------------

  // Prepare test histograms
  TH1F *hMExy[6];
  TH1F *hMETPhi[3];
  TH2F *hCaloMEx_LS;
  TH2F *hCaloMEy_LS;
  TH2F *hCaloMExNoHF_LS;
  TH2F *hCaloMEyNoHF_LS;

  if (RunDir == "") {
    newHistoName = "JetMET/CaloMETAnalyzer/METTask_";
  } else {
    newHistoName = RunDir+"/JetMET/Run summary/CaloMETAnalyzer/METTask_";
  }

  meNew = dbe->get(newHistoName+"CaloMEx");         if (meNew) hMExy[0] = meNew->getTH1F();
  meNew = dbe->get(newHistoName+"CaloMEy");         if (meNew) hMExy[1] = meNew->getTH1F();
  meNew = dbe->get(newHistoName+"CaloMExNoHF");     if (meNew) hMExy[2] = meNew->getTH1F();
  meNew = dbe->get(newHistoName+"CaloMEyNoHF");     if (meNew) hMExy[3] = meNew->getTH1F();
  meNew = dbe->get(newHistoName+"CaloMETPhi");      if (meNew) hMETPhi[0] = meNew->getTH1F();
  meNew = dbe->get(newHistoName+"CaloMETPhiNoHF");  if (meNew) hMETPhi[1] = meNew->getTH1F();

  meNew = dbe->get(newHistoName+"CaloMEx_LS");      if (meNew) hCaloMEx_LS     = meNew->getTH2F();
  meNew = dbe->get(newHistoName+"CaloMEy_LS");      if (meNew) hCaloMEy_LS     = meNew->getTH2F();
  meNew = dbe->get(newHistoName+"CaloMExNoHF_LS");  if (meNew) hCaloMExNoHF_LS = meNew->getTH2F();
  meNew = dbe->get(newHistoName+"CaloMEyNoHF_LS");  if (meNew) hCaloMEyNoHF_LS = meNew->getTH2F();

  // Prepare reference histograms
  TH1F *hRefMExy[6];
  TH1F *hRefMETPhi[2];

  if (RefRunDir == "") {
    refHistoName = "JetMET/CaloMETAnalyzer/METTask_";
  } else {
    refHistoName = RefRunDir+"/JetMET/Run summary/CaloMETAnalyzer/METTask_";
  }

  if (testType>=1){
    meRef = dbe->get(refHistoName+"CaloMEx");         if (meRef) hRefMExy[0] = meRef->getTH1F();
    meRef = dbe->get(refHistoName+"CaloMEy");         if (meRef) hRefMExy[1] = meRef->getTH1F();
    meRef = dbe->get(refHistoName+"CaloMExNoHF");     if (meRef) hRefMExy[2] = meRef->getTH1F();
    meRef = dbe->get(refHistoName+"CaloMEyNoHF");     if (meRef) hRefMExy[3] = meRef->getTH1F();
    meRef = dbe->get(refHistoName+"CaloMETPhi");      if (meRef) hRefMETPhi[0] = meRef->getTH1F();
    meRef = dbe->get(refHistoName+"CaloMETPhiNoHF");  if (meRef) hRefMETPhi[1] = meRef->getTH1F();
  }

  // 
  // Test 1D histograms
  //-------------------
  //
  // Prepare functions for fittings
  TF1 *g1    = new TF1("g1","gaus",-50,50);
  TF1 *g2    = new TF1("g2","gaus",-500,500);
  TF1 *dgaus = new TF1("dgaus","gaus(0)+gaus(3)",-500,500);

  TF1  *fitfun[6];
  TF1  *fitfun1[6];
  TF1  *fitfun2[6];

  if (METFIT){
    for (int i=0;i<4;i++) {
      if (hMExy[i]->GetSum()>0.){
	fitd(hMExy[i],dgaus,g1,g2,verbose);
	fitfun[i]  = hMExy[i]->GetFunction("dgaus");
	fitfun1[i] = (TF1*)g1->Clone();
	fitfun2[i] = (TF1*)g2->Clone();
      }
    }
  }

  // Chi2 test for 1D histograms
  //-----------------------------

  TH1F *h_ref;
  double fracErrorRef=0.3; // assign 30% error on reference histograms

  double test_METPhi=1.; // METPhi test values

  for (int i=0;i<4;i++) {

    h_ref = hRefMExy[i];
    // --- Adjust the reference histogram errors ---
    if (testType>=1){
      h_ref->Scale(hRefMExy[i]->GetEntries()/hRefMExy[i]->GetEntries());
      for (int ibin=0; ibin<h_ref->GetNbinsX(); ibin++){
	if (h_ref->GetBinContent(ibin+1)==0.){
	  h_ref->SetBinContent(ibin+1,1.);
	  h_ref->SetBinError(ibin+1,1.);
	} else if (h_ref->GetBinError(ibin+1)/h_ref->GetBinContent(ibin+1)<fracErrorRef) {	
	  h_ref->SetBinError(ibin+1,h_ref->GetBinContent(ibin+1)*fracErrorRef);
	}
      }
    } // testType>=1

    switch (testType) {
    case 1 :
      if (verbose) test_METPhi = hMExy[i]->KolmogorovTest(hRefMExy[i],"D");
      else         test_METPhi = hMExy[i]->KolmogorovTest(hRefMExy[i]);
      break;
    case 2 :
      if (verbose) test_METPhi = hMExy[i]->Chi2Test(h_ref,"UW,CHI2/NDF,P");
      else         test_METPhi = hMExy[i]->Chi2Test(h_ref,"UW,CHI2/NDF");
      break;
    }
    if (verbose > 0) std::cout << ">>> Test (" << testType 
			       << ") Result = " << test_METPhi << std::endl;    
  }

  for (int i=0;i<2;i++) {
    
    h_ref = hRefMETPhi[i];
    // --- Adjust the reference histogram errors ---
    if (testType>=1){
      h_ref->Scale(hMETPhi[i]->GetEntries()/hRefMETPhi[i]->GetEntries());
      for (int ibin=0; ibin<h_ref->GetNbinsX(); ibin++){
	if (h_ref->GetBinContent(ibin+1)==0.){
	  h_ref->SetBinContent(ibin+1,1.);
	  h_ref->SetBinError(ibin+1,1.);
	} else if (h_ref->GetBinError(ibin+1)/h_ref->GetBinContent(ibin+1)<fracErrorRef) {	
	  h_ref->SetBinError(ibin+1,h_ref->GetBinContent(ibin+1)*fracErrorRef);
	}
      }
    } // testType>=1

    switch (testType) {
    case 1 :
      if (verbose) test_METPhi = hMETPhi[i]->KolmogorovTest(hRefMETPhi[i],"D");
      else         test_METPhi = hMETPhi[i]->KolmogorovTest(hRefMETPhi[i]);
      break;
    case 2 :
      if (verbose) test_METPhi = hMETPhi[i]->Chi2Test(h_ref,"UW,CHI2/NDF,P");
      else         test_METPhi = hMETPhi[i]->Chi2Test(h_ref,"UW,CHI2/NDF");      
      break;
    }
    if (verbose > 0) std::cout << ">>> Test (" << testType 
			       << ") Result = " << test_METPhi << std::endl;    
  }

  // 
  // Test 2D histograms
  //-------------------

  // Slice *_LS histograms
  TH1D *CaloMEx_LS[nLSBins];
  TH1D *CaloMEy_LS[nLSBins];
  TH1D *CaloMExNoHF_LS[nLSBins];
  TH1D *CaloMEyNoHF_LS[nLSBins];
  TF1 *fitfun_CaloMEx_LS[nLSBins];
  TF1 *fitfun_CaloMEy_LS[nLSBins];
  TF1 *fitfun_CaloMExNoHF_LS[nLSBins];
  TF1 *fitfun_CaloMEyNoHF_LS[nLSBins];
  TF1 *fitfun1_CaloMEx_LS[nLSBins];
  TF1 *fitfun1_CaloMEy_LS[nLSBins];
  TF1 *fitfun1_CaloMExNoHF_LS[nLSBins];
  TF1 *fitfun1_CaloMEyNoHF_LS[nLSBins];
  TF1 *fitfun2_CaloMEx_LS[nLSBins];
  TF1 *fitfun2_CaloMEy_LS[nLSBins];
  TF1 *fitfun2_CaloMExNoHF_LS[nLSBins];
  TF1 *fitfun2_CaloMEyNoHF_LS[nLSBins];
  int JetMET_MET[nLSBins];
  int JetMET_MET_All[nLSBins];
  int JetMET_MEx_All[nLSBins];
  int JetMET_MEy_All[nLSBins];
  int JetMET_MET_NoHF[nLSBins];
  int JetMET_MEx_NoHF[nLSBins];
  int JetMET_MEy_NoHF[nLSBins];
  for (int i=0;i<nLSBins;i++){
    JetMET_MET[i]     =-1;
    JetMET_MET_All[i] =-1;
    JetMET_MEx_All[i] =-1;
    JetMET_MEy_All[i] =-1;
    JetMET_MET_NoHF[i]=-1;
    JetMET_MEx_NoHF[i]=-1;
    JetMET_MEy_NoHF[i]=-1;
  }
  char ctitle[100];

  // (LS=0 assigned to the entire run)
  for (int LS=1; LS<nLSBins; LS++){

    // Projection returns a 
    sprintf(ctitle,"CaloMEx_%04d",LS);     CaloMEx_LS[LS]=hCaloMEx_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMEy_%04d",LS);     CaloMEy_LS[LS]=hCaloMEy_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMExNoHF_%04d",LS); CaloMExNoHF_LS[LS]=hCaloMExNoHF_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMEyNoHF_%04d",LS); CaloMEyNoHF_LS[LS]=hCaloMEyNoHF_LS->ProjectionX(ctitle,LS+1,LS+1);

    if (METFIT){
      if (CaloMEx_LS[LS]->GetSum()>0.) {
	fitdd(CaloMEx_LS[LS],dgaus,g1,g2,verbose);
        fitfun_CaloMEx_LS[LS]=CaloMEx_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMEx_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMEx_LS[LS]=(TF1*)g2->Clone();
      }
      if (CaloMEy_LS[LS]->GetSum()>0.) {
	fitdd(CaloMEy_LS[LS],dgaus,g1,g2,verbose);
        fitfun_CaloMEy_LS[LS]=CaloMEy_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMEy_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMEy_LS[LS]=(TF1*)g2->Clone();
      }
      if (CaloMExNoHF_LS[LS]->GetSum()>0.) {
	fitdd(CaloMExNoHF_LS[LS],dgaus,g1,g2,verbose);
        fitfun_CaloMExNoHF_LS[LS]=CaloMExNoHF_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMExNoHF_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMExNoHF_LS[LS]=(TF1*)g2->Clone();
      }
      if (CaloMEyNoHF_LS[LS]->GetSum()>0.) {
	fitdd(CaloMEyNoHF_LS[LS],dgaus,g1,g2,verbose);
        fitfun_CaloMEyNoHF_LS[LS]=CaloMEyNoHF_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMEyNoHF_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMEyNoHF_LS[LS]=(TF1*)g2->Clone();
      }
    } // METFIT

  }   // loop over LS

  //----------------------------------------------------------------
  //--- Apply data certification algorithm
  //----------------------------------------------------------------

  double chisq_threshold_run     = 500.;
  double chisq_threshold_lumisec = 20.;
  double MEx_threshold   = 10.;
  double MEy_threshold   = 10.;
  double MExRMS_threshold   = 10.;
  double MEyRMS_threshold   = 10.;
  int    minEntry        = 5;

  if (verbose) {
    std::cout << std::endl;
    if (METFIT)
      printf("| Variable                       |   Reduced chi^2              | Mean               | Width      |\n");
    else 
      printf("| Variable                       |   Mean   | RMS   |\n");
  }
  //
  // Entire run
  //-----------------------------------
  for (int i=0;i<4;i++){
    if (METFIT) { //---------- MET fits ----------
      int nmean=1;
      if (fitfun[i]->GetNumberFreeParameters()==3) nmean=4;
      if (verbose)
	printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
	       hMExy[i]->GetName(),
	       fitfun[i]->GetChisquare(),double(fitfun[i]->GetNDF()),
	       fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
	       fitfun[i]->GetParameter(nmean),  fitfun[i]->GetParError(nmean+1),
	       fitfun[i]->GetParameter(nmean+1),fitfun[i]->GetParError(nmean+1));
      if (i==0)
	JetMET_MEx_All[0]=data_certificate_metfit(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
					       fitfun[i]->GetParameter(nmean),
					       chisq_threshold_run,MEx_threshold);
      if (i==1)
	JetMET_MEy_All[0]=data_certificate_metfit(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
					       fitfun[i]->GetParameter(nmean),
					       chisq_threshold_run,MEy_threshold);
      if (i==2)
	JetMET_MEx_NoHF[0]=data_certificate_metfit(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
						fitfun[i]->GetParameter(nmean),
						chisq_threshold_run,MEx_threshold);
      if (i==3)
	JetMET_MEy_NoHF[0]=data_certificate_metfit(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
						fitfun[i]->GetParameter(nmean),
						chisq_threshold_run,MEy_threshold);
    } else { //----------- no MET fits ----------
      if (verbose)
	printf("| %-30s | %8.3f | %8.3f |\n",hMExy[i]->GetName(),hMExy[i]->GetMean(),hMExy[i]->GetRMS());
      if (i==0)
	JetMET_MEx_All[0]=data_certificate_met(hMExy[i]->GetMean(),hMExy[i]->GetRMS(),
					       MEx_threshold,MExRMS_threshold);
      if (i==1)
	JetMET_MEy_All[0]=data_certificate_met(hMExy[i]->GetMean(),hMExy[i]->GetRMS(),
					       MEy_threshold,MEyRMS_threshold);
      if (i==2)
	JetMET_MEx_NoHF[0]=data_certificate_met(hMExy[i]->GetMean(),hMExy[i]->GetRMS(),
						MEx_threshold,MExRMS_threshold);
      if (i==3)
	JetMET_MEy_NoHF[0]=data_certificate_met(hMExy[i]->GetMean(),hMExy[i]->GetRMS(),
						MEy_threshold,MEyRMS_threshold);      
    }
  }

  //
  // Each lumi section
  // (LS=0 assigned to the entire run)
  //-----------------------------------
  for (int LS=1; LS<500; LS++){

    //----- CaloMEx ------------------------------
    if (CaloMEx_LS[LS]->GetSum()>0.) {

      if (METFIT) { //---------- MET fits ----------
	int nmean=1;
	if (fitfun_CaloMEx_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
	if (verbose)
	  printf("\n| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
		 CaloMEx_LS[LS]->GetName(),
		 fitfun_CaloMEx_LS[LS]->GetChisquare(),double(fitfun_CaloMEx_LS[LS]->GetNDF()),
		 fitfun_CaloMEx_LS[LS]->GetChisquare()/double(fitfun_CaloMEx_LS[LS]->GetNDF()),
		 fitfun_CaloMEx_LS[LS]->GetParameter(nmean),  fitfun_CaloMEx_LS[LS]->GetParError(nmean),
		 fitfun_CaloMEx_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEx_LS[LS]->GetParError(nmean+1));
	JetMET_MEx_All[LS]=data_certificate_metfit(fitfun_CaloMEx_LS[LS]->GetChisquare()/double(fitfun_CaloMEx_LS[LS]->GetNDF()),
						   fitfun_CaloMEx_LS[LS]->GetParameter(nmean),
						   chisq_threshold_lumisec,MEx_threshold);
      } else { //----------- no MET fits ----------
	if (verbose)
	printf("| %-30s | %8.3f | %8.3f |\n",CaloMEx_LS[LS]->GetName(),CaloMEx_LS[LS]->GetMean(),CaloMEx_LS[LS]->GetRMS());
	JetMET_MEx_All[LS]=data_certificate_met(CaloMEx_LS[LS]->GetMean(),CaloMEx_LS[LS]->GetRMS(),MEx_threshold,MExRMS_threshold);
      }

    }
    //----- CaloMEy ------------------------------
    if (CaloMEy_LS[LS]->GetSum()>0.) {

      if (METFIT) {
      int nmean=1;
      if (fitfun_CaloMEy_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      if (verbose)
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEy_LS[LS]->GetName(),
             fitfun_CaloMEy_LS[LS]->GetChisquare(),double(fitfun_CaloMEy_LS[LS]->GetNDF()),
             fitfun_CaloMEy_LS[LS]->GetChisquare()/double(fitfun_CaloMEy_LS[LS]->GetNDF()),
             fitfun_CaloMEy_LS[LS]->GetParameter(nmean),  fitfun_CaloMEy_LS[LS]->GetParError(nmean),
             fitfun_CaloMEy_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEy_LS[LS]->GetParError(nmean+1));
      JetMET_MEy_All[LS]=data_certificate_metfit(fitfun_CaloMEy_LS[LS]->GetChisquare()/double(fitfun_CaloMEy_LS[LS]->GetNDF()),
                                          fitfun_CaloMEy_LS[LS]->GetParameter(nmean),
                                          chisq_threshold_lumisec,MEy_threshold);
      } else { //----------- no MET fits ----------
	if (verbose)
	printf("| %-30s | %8.3f | %8.3f |\n",CaloMEy_LS[LS]->GetName(),CaloMEy_LS[LS]->GetMean(),CaloMEy_LS[LS]->GetRMS());
	JetMET_MEy_All[LS]=data_certificate_met(CaloMEy_LS[LS]->GetMean(),CaloMEy_LS[LS]->GetRMS(),MEy_threshold,MEyRMS_threshold);
      }
      
    }

    //----- CaloMExNoHF ------------------------------
    if (CaloMExNoHF_LS[LS]->GetSum()>0.) {

      if (METFIT) {
	int nmean=1;
	if (fitfun_CaloMExNoHF_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
	if (verbose)
	  printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
		 CaloMExNoHF_LS[LS]->GetName(),
		 fitfun_CaloMExNoHF_LS[LS]->GetChisquare(),double(fitfun_CaloMExNoHF_LS[LS]->GetNDF()),
		 fitfun_CaloMExNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMExNoHF_LS[LS]->GetNDF()),
		 fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean),  fitfun_CaloMExNoHF_LS[LS]->GetParError(nmean),
		 fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean+1),fitfun_CaloMExNoHF_LS[LS]->GetParError(nmean+1));
	JetMET_MEx_NoHF[LS]=data_certificate_metfit(fitfun_CaloMExNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMExNoHF_LS[LS]->GetNDF()),
						    fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean),
						    chisq_threshold_lumisec,MEx_threshold);
      } else { //----------- no MET fits ----------
	if (verbose)
	printf("| %-30s | %8.3f | %8.3f |\n",CaloMExNoHF_LS[LS]->GetName(),CaloMExNoHF_LS[LS]->GetMean(),CaloMExNoHF_LS[LS]->GetRMS());
	JetMET_MEx_NoHF[LS]=data_certificate_met(CaloMExNoHF_LS[LS]->GetMean(),CaloMExNoHF_LS[LS]->GetRMS(),
						 MEx_threshold,MExRMS_threshold);
      }
      
    }

    //----- CaloMEyNoHF ------------------------------
    if (CaloMEyNoHF_LS[LS]->GetSum()>0.) {

      if (METFIT) {
	int nmean=1;
	if (fitfun_CaloMEyNoHF_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
	if (verbose)
	  printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
		 CaloMEyNoHF_LS[LS]->GetName(),
		 fitfun_CaloMEyNoHF_LS[LS]->GetChisquare(),double(fitfun_CaloMEyNoHF_LS[LS]->GetNDF()),
		 fitfun_CaloMEyNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMEyNoHF_LS[LS]->GetNDF()),
		 fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean),  fitfun_CaloMEyNoHF_LS[LS]->GetParError(nmean),
		 fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEyNoHF_LS[LS]->GetParError(nmean+1));
	JetMET_MEy_NoHF[LS]=data_certificate_metfit(fitfun_CaloMEyNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMEyNoHF_LS[LS]->GetNDF()),
						    fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean),
						    chisq_threshold_lumisec,MEy_threshold);
      }  else { //----------- no MET fits ----------
	if (verbose)
	printf("| %-30s | %8.3f | %8.3f |\n",CaloMEyNoHF_LS[LS]->GetName(),CaloMEyNoHF_LS[LS]->GetMean(),CaloMEyNoHF_LS[LS]->GetRMS());
	JetMET_MEy_NoHF[LS]=data_certificate_met(CaloMEyNoHF_LS[LS]->GetMean(),CaloMEyNoHF_LS[LS]->GetRMS(),
						 MEy_threshold,MEyRMS_threshold);
      }      

    }
  } // loop over LS

  //
  // Final MET data certification algorithm
  //----------------------------------------
  for (int LS=0; LS<nLSBins; LS++){
    JetMET_MET_All[LS] = JetMET_MEx_All[LS] * JetMET_MEy_All[LS];
    JetMET_MET_NoHF[LS]= JetMET_MEx_NoHF[LS]* JetMET_MEy_NoHF[LS];
    JetMET_MET[LS]     = JetMET_MET_All[LS] * JetMET_MET_NoHF[LS];

    // -- Fill the DC Result Histograms (entire run)
    if (LS==0){
    mMETDCFL1->Fill(double(JetMET_MET[LS]));
    mMETDCFL3[0]->Fill(double(JetMET_MET_All[LS]));
    mMETDCFL3[1]->Fill(double(JetMET_MET_NoHF[LS]));
    }

  }

  //
  // Final MET data certification algorithm
  //----------------------------------------
  if (DEBUG) {
  std::cout << std::endl;
  printf("   run,       lumi-sec, tag name,                                   output\n");
  int LS_LAST=-1;
  for (int LS=0; LS<nLSBins; LS++){

    if (LS==0){                                                                              // For entire run,
      printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET",     JetMET_MET[LS]);         // always print out.
      printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
      printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
    } else {
      if (CaloMEx_LS[LS]->GetSum()>minEntry) {                                               // Lumi Section with data
	if (LS_LAST==-1){                                                                    // For first lumi section,
	  printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET",     JetMET_MET[LS]);     // always print out.
	  printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
	  printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
	}
	else {
	  if ( (JetMET_MET[LS]!=JetMET_MET[LS_LAST]) ||                                      // If changed from the previous lumi section
	       (JetMET_MET_All[LS]!=JetMET_MET_All[LS_LAST]) ||
	       (JetMET_MET_NoHF[LS]!=JetMET_MET_NoHF[LS_LAST]) ){
	    printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET",     JetMET_MET[LS]);
	    printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
	    printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
	  }
	}
	LS_LAST=LS;
      }    
    }      // LS==0
  }        // for LS
  std::cout << std::endl;
  }

  // -- 
  //dbe->rmdir(RefRunDir); // Delete reference plots from DQMStore
  // --
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DataCertificationJetMET::endJob() {

  //  LogTrace(metname)<<"[DataCertificationJetMET] Saving the histos";
  //  bool outputFile            = conf_.getParameter<bool>("OutputFile");
  //  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");

  bool outputFile            = conf_.getUntrackedParameter<bool>("OutputFile");
  std::string outputFileName = conf_.getUntrackedParameter<std::string>("OutputFileName");

  if (DEBUG) std::cout << ">>> endJob " << outputFile << std:: endl;

  if(outputFile){
    //dbe->showDirStructure();
    dbe->save(outputFileName);
  }
}

// ------------------------------------------------------------
int 
DataCertificationJetMET::data_certificate_metfit(double chi2, double mean, double chi2_tolerance, double mean_tolerance){
  int value=0;
  if (chi2<chi2_tolerance && fabs(mean)<mean_tolerance) value=1;
  return value;
}

// ------------------------------------------------------------
int 
DataCertificationJetMET::data_certificate_met(double mean, double rms, double mean_tolerance, double rms_tolerance){
  int value=0;
  if (rms<rms_tolerance && fabs(mean)<mean_tolerance) value=1;
  return value;
}

// ------------------------------------------------------------
void
DataCertificationJetMET::fitd(TH1F* hist, TF1* fn, TF1* f1, TF1* f2, int verbose){
  //
  Option_t *fit_option;
  fit_option = "QR0";
  if      (verbose==0) fit_option = "R0";
  else if (verbose==1) fit_option = "VR0";
  //
  Double_t par[6];
  Double_t pare[6];
  for (int i=0;i<6;i++){
    par[i] =0.;
    pare[i]=1.;
  }
  //
  //hist->GetXaxis()->SetRange(201,300);
  //
  //
  // First, single Gaussian fit
//   hist->Fit(f2,"RV")
//   f2->GetParameters(&par[3]);
//   fn->SetParameters(par);
//   fn->FixParameter(0,0.);
//   fn->FixParameter(1,0.);
//   fn->FixParameter(2,0.);
//   fn->SetParLimits(5,0.,1000.);
//   fn->SetParName(3,"Constant");
//   fn->SetParName(4,"Mean");
//   fn->SetParName(5,"Sigma");
//   hist->Fit(fn,"RV");  
  //
  //
  // Second, double Gaussian fit
  double chi2=4.;
  //if (fn->GetNDF()>0.) chi2=fn->GetChisquare()/fn->GetNDF();
  if (chi2>3.){
    hist->Fit(f1,fit_option);
    f1->GetParameters(&par[0]);
    fn->SetParameters(par);
    fn->ReleaseParameter(0);
    fn->ReleaseParameter(1);
    fn->ReleaseParameter(2);
    fn->SetParLimits(2,0.,1000.);
    fn->SetParName(0,"Constant");
    fn->SetParName(1,"Mean");
    fn->SetParName(2,"Sigma");
    fn->SetParName(3,"Constant2");
    fn->SetParName(4,"Mean2");
    fn->SetParName(5,"Sigma2");
    fn->SetParameter(5,par[2]*10.);
    hist->Fit(fn,fit_option);
    fn->GetParameters(&par[0]);
    pare[2]=fn->GetParError(2);
    pare[5]=fn->GetParError(5);
    f1->SetParameter(0,par[0]);
    f1->SetParameter(1,par[1]);
    f1->SetParameter(2,par[2]);
    f2->SetParameter(0,par[3]);
    f2->SetParameter(1,par[4]);
    f2->SetParameter(2,par[5]);   
  } 
  //
  //
  // Third, if two Gaussians have very similar widths,
  // set the initial value for the 2nd one to ~ x10 larger
//   if ( fabs(par[2]-par[5])<sqrt(pow(pare[2],2)+pow(pare[5],2)) ){
//     fn->SetParameter(5,par[2]*10.);
//     hist->Fit(fn,fit_option);
//     fn->GetParameters(&par[0]);
//     f1->SetParameter(0,par[0]);
//     f1->SetParameter(1,par[1]);
//     f1->SetParameter(2,par[2]);
//     f2->SetParameter(0,par[3]);
//     f2->SetParameter(1,par[4]);
//     f2->SetParameter(2,par[5]);   
//   } 
  //
  //
  // Fourth, if two Gaussians still have very similar widths,
  // set the initial value for the 2nd one to ~ x100 larger
  if ( fabs(par[2]-par[5])<sqrt(pow(pare[2],2)+pow(pare[5],2)) ){
    fn->SetParameter(5,par[2]*100.);
    hist->Fit(fn,fit_option);
    fn->GetParameters(&par[0]);
    f1->SetParameter(0,par[0]);
    f1->SetParameter(1,par[1]);
    f1->SetParameter(2,par[2]);
    f2->SetParameter(0,par[3]);
    f2->SetParameter(1,par[4]);
    f2->SetParameter(2,par[5]);   
  } 
  //
}

// ------------------------------------------------------------
void
DataCertificationJetMET::fitdd(TH1D* hist, TF1* fn, TF1* f1, TF1* f2, int verbose){
  //
  Option_t *fit_option;
  fit_option = "QR0";
  if      (verbose==0) fit_option = "R0";
  else if (verbose==1) fit_option = "VR0";
  //
  Double_t par[6];
  Double_t pare[6];
  for (int i=0;i<6;i++){
    par[i] =0.;
    pare[i]=1.;
  }
  //
  //hist->GetXaxis()->SetRange(201,300);
  //
  //
  // First, single Gaussian fit
//   hist->Fit(f2,fit_option)
//   f2->GetParameters(&par[3]);
//   fn->SetParameters(par);
//   fn->FixParameter(0,0.);
//   fn->FixParameter(1,0.);
//   fn->FixParameter(2,0.);
//   fn->SetParLimits(5,0.,1000.);
//   fn->SetParName(3,"Constant");
//   fn->SetParName(4,"Mean");
//   fn->SetParName(5,"Sigma");
//   hist->Fit(fn,"RV");  
  //
  //
  // Second, double Gaussian fit
  double chi2=4.;
  //if (fn->GetNDF()>0.) chi2=fn->GetChisquare()/fn->GetNDF();
  if (chi2>3.){
    hist->Fit(f1,fit_option);
    f1->GetParameters(&par[0]);
    fn->SetParameters(par);
    fn->ReleaseParameter(0);
    fn->ReleaseParameter(1);
    fn->ReleaseParameter(2);
    fn->SetParLimits(2,0.,1000.);
    fn->SetParName(0,"Constant");
    fn->SetParName(1,"Mean");
    fn->SetParName(2,"Sigma");
    fn->SetParName(3,"Constant2");
    fn->SetParName(4,"Mean2");
    fn->SetParName(5,"Sigma2");
    fn->SetParameter(5,par[2]*10.);
    hist->Fit(fn,fit_option);
    fn->GetParameters(&par[0]);
    pare[2]=fn->GetParError(2);
    pare[5]=fn->GetParError(5);
    f1->SetParameter(0,par[0]);
    f1->SetParameter(1,par[1]);
    f1->SetParameter(2,par[2]);
    f2->SetParameter(0,par[3]);
    f2->SetParameter(1,par[4]);
    f2->SetParameter(2,par[5]);   
  } 
  //
  //
  // Third, if two Gaussians have very similar widths,
  // set the initial value for the 2nd one to ~ x10 larger
//   if ( fabs(par[2]-par[5])<sqrt(pow(pare[2],2)+pow(pare[5],2)) ){
//     fn->SetParameter(5,par[2]*10.);
//     std::cout << "aaa3" << std::endl;
//     hist->Fit(fn,fit_option);
//     fn->GetParameters(&par[0]);
//     f1->SetParameter(0,par[0]);
//     f1->SetParameter(1,par[1]);
//     f1->SetParameter(2,par[2]);
//     f2->SetParameter(0,par[3]);
//     f2->SetParameter(1,par[4]);
//     f2->SetParameter(2,par[5]);   
//   } 
  //
  //
  // Fourth, if two Gaussians still have very similar widths,
  // set the initial value for the 2nd one to ~ x100 larger
  if ( fabs(par[2]-par[5])<sqrt(pow(pare[2],2)+pow(pare[5],2)) ){
    fn->SetParameter(5,par[2]*100.);
    hist->Fit(fn,fit_option);
    fn->GetParameters(&par[0]);
    f1->SetParameter(0,par[0]);
    f1->SetParameter(1,par[1]);
    f1->SetParameter(2,par[2]);
    f2->SetParameter(0,par[3]);
    f2->SetParameter(1,par[4]);
    f2->SetParameter(2,par[5]);   
  } 
  //
}

//define this as a plug-in
DEFINE_FWK_MODULE(DataCertificationJetMET);
