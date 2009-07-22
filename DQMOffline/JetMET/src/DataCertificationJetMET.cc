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
// $Id: DataCertificationJetMET.cc,v 1.24 2009/03/28 00:19:58 hatake Exp $
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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

// Some switches
#define NJetAlgo 4
#define NL3Flags 3

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

      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;

      virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
      virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   DQMStore * dbe;
   edm::Service<TFileService> fs_;

   int verbose;


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
DataCertificationJetMET::beginJob(const edm::EventSetup& c)
{

  // -----------------------------------------
  // verbose 0: suppress printouts
  //         1: show printouts
  verbose   = conf_.getUntrackedParameter<int>("Verbose");

  if (verbose) std::cout << ">>> BeginJob (DataCertificationJetMET) <<<" << std::endl;

}

// ------------ method called once each job after finishing event loop  ------------
void 
DataCertificationJetMET::endJob()
{

  if (verbose) std::cout << ">>> EndJob (DataCertificationJetMET) <<<" << std::endl;

  bool outputFile            = conf_.getUntrackedParameter<bool>("OutputFile");
  std::string outputFileName = conf_.getUntrackedParameter<std::string>("OutputFileName");
  if (verbose) std::cout << ">>> endJob " << outputFile << std:: endl;

  if(outputFile){
    //dbe->showDirStructure();
    dbe->save(outputFileName);
  }

}
 
// ------------ method called just before starting a new lumi section  ------------
void 
DataCertificationJetMET::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& c)
{

  if (verbose) std::cout << ">>> BeginLuminosityBlock (DataCertificationJetMET) <<<" << std::endl;
  if (verbose) std::cout << ">>> lumiBlock = " << lumiBlock.id()                   << std::endl;
  if (verbose) std::cout << ">>> run       = " << lumiBlock.id().run()             << std::endl;
  if (verbose) std::cout << ">>> lumiBlock = " << lumiBlock.id().luminosityBlock() << std::endl;

}

// ------------ method called just after a lumi section ends  ------------
void 
DataCertificationJetMET::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& c)
{

  if (verbose) std::cout << ">>> EndLuminosityBlock (DataCertificationJetMET) <<<" << std::endl;
  if (verbose) std::cout << ">>> lumiBlock = " << lumiBlock.id()                   << std::endl;
  if (verbose) std::cout << ">>> run       = " << lumiBlock.id().run()             << std::endl;
  if (verbose) std::cout << ">>> lumiBlock = " << lumiBlock.id().luminosityBlock() << std::endl;

  dbe = edm::Service<DQMStore>().operator->();    
  dbe->setCurrentFolder("JetMET");

  //
  //-----
  MonitorElement * meMETPhi=0;
  meMETPhi = new MonitorElement(*(dbe->get("JetMET/MET/CaloMET/METTask_CaloMETPhi")));
  const QReport * myQReport = meMETPhi->getQReport("phiQTest"); //get QReport associated to your ME  
  if(myQReport) {
    float qtresult = myQReport->getQTresult(); // get QT result value
    int qtstatus   = myQReport->getStatus() ;  // get QT status value (see table below)
    std::string qtmessage = myQReport->getMessage() ; // get the whole QT result message
    if (verbose) std::cout << "test" << qtmessage << " qtresult = " << qtresult << " qtstatus = " << qtstatus << std::endl;    
  }

}

// ------------ method called just before starting a new run  ------------
void 
DataCertificationJetMET::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  if (verbose) std::cout << ">>> BeginRun (DataCertificationJetMET) <<<" << std::endl;
  if (verbose) std::cout << ">>> run = " << run.id() << std::endl;

}

// ------------ method called right after a run ends ------------
void 
DataCertificationJetMET::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  
  if (verbose) std::cout << ">>> EndRun (DataCertificationJetMET) <<<" << std::endl;
  if (verbose) std::cout << ">>> run = " << run.id() << std::endl;

  // -----------------------------------------
  // testType 0: no comparison with histograms
  //          1: KS test
  //          2: Chi2 test
  //
  int testType = 0; 
  testType  = conf_.getUntrackedParameter<int>("TestType");
  if (verbose) std::cout << ">>> TestType        = " <<  testType  << std::endl;  

  std::vector<MonitorElement*> mes;
  std::vector<std::string> subDirVec;
  std::string RunDir;
  std::string RunNum;
  int         RunNumber;
  std::string RefRunDir;
  std::string RefRunNum;
  int         RefRunNumber;
    
  std::string filename    = conf_.getUntrackedParameter<std::string>("fileName");
  if (verbose) std::cout << ">>> FileName        = " << filename    << std::endl;
  bool InMemory = true;

  if (filename != "") InMemory = false;
  if (verbose) std::cout << "InMemory           = " << InMemory    << std::endl;

  if (InMemory) {
    //----------------------------------------------------------------
    // Histograms are in memory (for standard full-chain mode)
    //----------------------------------------------------------------

    dbe = edm::Service<DQMStore>().operator->();
    //dbe->showDirStructure();
    
    mes = dbe->getAllContents("");
    if (verbose) std::cout << "1 >>> found " <<  mes.size() << " monitoring elements!" << std::endl;

    dbe->setCurrentFolder("JetMET");
    subDirVec = dbe->getSubdirs();

    for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {    
      if (verbose) std::cout << "-AAA- Dir = >>" << ic->c_str() << "<<" << std::endl;
    }

    RunDir    = "";
    RefRunDir = "";

    RunNumber = run.id().run();

  } else {
    //----------------------------------------------------------------
    // Open input files (for standalone mode)
    //----------------------------------------------------------------

    std::string filename    = conf_.getUntrackedParameter<std::string>("fileName");
    if (verbose) std::cout << "FileName           = " << filename    << std::endl;

    std::string reffilename;
    if (testType>=1){
      reffilename = conf_.getUntrackedParameter<std::string>("refFileName");
      if (verbose) std::cout << "Reference FileName = " << reffilename << std::endl;
    }

    // -- Current & Reference Run
    //---------------------------------------------
    dbe = edm::Service<DQMStore>().operator->();
    dbe->open(filename);
    if (testType>=1) dbe->open(reffilename);

    mes = dbe->getAllContents("");
    if (verbose) std::cout << "found " <<  mes.size() << " monitoring elements!" << std::endl;
    
    dbe->setCurrentFolder("/");
    std::string currDir = dbe->pwd();
    if (verbose) std::cout << "--- Current Directory " << currDir << std::endl;

    subDirVec = dbe->getSubdirs();

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
      if (verbose) std::cout << "-XXX- Dir = >>" << ic->c_str() << "<<" << std::endl;
      ind++;
    }

    //
    // Current
    //
    if (RunDir == "JetMET") {
      RunDir = "";
      if (verbose) std::cout << "-XXX- RunDir = >>" << RunDir.c_str() << "<<" << std::endl;
    }
    RunNum.erase(0,4);
    RunNumber = atoi(RunNum.c_str());
    if (verbose) std::cout << "--- >>" << RunNumber << "<<" << std::endl;

    //
    // Reference
    //
    if (testType>=1){
      
      if (RefRunDir == "JetMET") {
	RefRunDir = "";
	if (verbose) std::cout << "-XXX- RefRunDir = >>" << RefRunDir.c_str() << "<<" << std::endl;
      }
      RefRunNum.erase(0,4);
      RefRunNumber = atoi(RefRunNum.c_str());
      if (verbose) std::cout << "--- >>" << RefRunNumber << "<<" << std::endl;
      
    }
    //  ic++;
  }


  //----------------------------------------------------------------
  // Reference Histograms
  //----------------------------------------------------------------


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

  if (verbose) std::cout << RunDir << std::endl;
  dbe->setCurrentFolder("JetMET/EventInfo/Certification/");    

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
        refHistoName = "JetMET/Jet/IterativeConeJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/Jet/IterativeConeJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/Jet/IterativeConeJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/Jet/IterativeConeJets/";
      }
    }
    if (iAlgo == 1) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/Jet/SISConeJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/Jet/SISConeJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/Jet/SISConeJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/Jet/SISConeJets/";
      }
    }
    if (iAlgo == 2) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/Jet/PFJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/Jet/PFJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/Jet/PFJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/Jet/PFJets/";
      }
    }
    if (iAlgo == 3) {
      if (RefRunDir == "") {
        refHistoName = "JetMET/Jet/JPTJets/";
      } else {
        refHistoName = RefRunDir+"/JetMET/Run summary/Jet/JPTJets/";
      }
      if (RunDir == "") {
        newHistoName = "JetMET/Jet/JPTJets/";
      } else {
        newHistoName = RunDir+"/JetMET/Run summary/Jet/JPTJets/";
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
  if (verbose) {
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

  //
  //-----
//   std::cout << "aaa " << std::endl;
//   MonitorElement * meMETPhi=0;
//   meMETPhi = new MonitorElement(*(dbe->get("JetMET/CaloMETAnalyzer/METTask_CaloMETPhi")));
//   const QReport * myQReport = meMETPhi->getQReport("phiQTest"); //get QReport associated to your ME  
//   std::cout << "aaa " << myQReport << std::endl;    
//   if(myQReport) {
//     float qtresult = myQReport->getQTresult(); // get QT result value
//     int qtstatus   = myQReport->getStatus() ;  // get QT status value (see table below)
//     std::string qtmessage = myQReport->getMessage() ; // get the whole QT result message
//     std::cout << "aaa" << qtmessage << " " << qtresult << " " << qtstatus << std::endl;    
//   }

  //
  // Prepare test histograms
  //
  MonitorElement *meMExy[6];
  MonitorElement *meMETPhi[3];

  if (RunDir == "") {
    newHistoName = "JetMET/MET/";
  } else {
    newHistoName = RunDir+"/JetMET/Run summary/MET/";
  }

  meMExy[0]   = new MonitorElement(*(dbe->get((newHistoName+"CaloMET/METTask_CaloMEx"))));
  meMExy[1]   = new MonitorElement(*(dbe->get((newHistoName+"CaloMET/METTask_CaloMEy"))));
  meMExy[2]   = new MonitorElement(*(dbe->get((newHistoName+"CaloMETNoHF/METTask_CaloMEx"))));
  meMExy[3]   = new MonitorElement(*(dbe->get((newHistoName+"CaloMETNoHF/METTask_CaloMEy"))));
  meMETPhi[0] = new MonitorElement(*(dbe->get((newHistoName+"CaloMET/METTask_CaloMETPhi"))));
  meMETPhi[1] = new MonitorElement(*(dbe->get((newHistoName+"CaloMETNoHF/METTask_CaloMETPhi"))));
				   
  //----------------------------------------------------------------
  //--- Extract quality test results and fill data certification results
  //----------------------------------------------------------------

  const QReport * QReport_MExy[6];
  const QReport * QReport_METPhi[3];
  float qr_JetMET_MExy[6];
  float qr_JetMET_METPhi[3];
  float dc_JetMET[3];

  QReport_METPhi[0] = meMETPhi[0]->getQReport("phiQTest"); //get QReport associated to this ME  
  QReport_METPhi[1] = meMETPhi[1]->getQReport("phiQTest"); //get QReport associated to this ME

  QReport_MExy[0] = meMExy[0]->getQReport("meanQTest"); //get QReport associated to this ME  
  QReport_MExy[1] = meMExy[1]->getQReport("meanQTest"); //get QReport associated to this ME
  QReport_MExy[2] = meMExy[2]->getQReport("meanQTest"); //get QReport associated to this ME  
  QReport_MExy[3] = meMExy[3]->getQReport("meanQTest"); //get QReport associated to this ME

  if (QReport_MExy[0]){
    if (QReport_MExy[0]->getStatus()==100) 
      qr_JetMET_MExy[0] = QReport_MExy[0]->getQTresult();
    if (verbose) std::cout << QReport_MExy[0]->getMessage() << std::endl;
  }

  if (QReport_MExy[1]){
    if (QReport_MExy[1]->getStatus()==100) 
      qr_JetMET_MExy[1] = QReport_MExy[1]->getQTresult();
    if (verbose) std::cout << QReport_MExy[1]->getMessage() << std::endl;
  }

  if (QReport_MExy[2]){
    if (QReport_MExy[2]->getStatus()==100) 
      qr_JetMET_MExy[2] = QReport_MExy[2]->getQTresult();
    if (verbose) std::cout << QReport_MExy[2]->getMessage() << std::endl;
  }

  if (QReport_MExy[3]){
    if (QReport_MExy[3]->getStatus()==100) 
      qr_JetMET_MExy[3] = QReport_MExy[3]->getQTresult();
    if (verbose) std::cout << QReport_MExy[3]->getMessage() << std::endl;
  }

  if (QReport_METPhi[0]){
    if (QReport_METPhi[0]->getStatus()==100) 
      qr_JetMET_METPhi[0] = QReport_METPhi[0]->getQTresult();
    if (verbose) std::cout << QReport_METPhi[0]->getMessage() << std::endl;
  }

  if (QReport_METPhi[1]){
    if (QReport_METPhi[1]->getStatus()==100) 
      qr_JetMET_METPhi[1] = QReport_METPhi[1]->getQTresult();
    if (verbose) std::cout << QReport_METPhi[1]->getMessage() << std::endl;
  }

  dc_JetMET[0]=0.;
//   if (qr_JetMET_MExy[0]*qr_JetMET_MExy[1]*qr_JetMET_METPhi[0] > 0.5 )
  if (qr_JetMET_MExy[0]*qr_JetMET_MExy[1] > 0.5 )
    dc_JetMET[0]=1.;

  dc_JetMET[1]=0.;
//   if (qr_JetMET_MExy[2]*qr_JetMET_MExy[3]*qr_JetMET_METPhi[1] > 0.5 )
  if (qr_JetMET_MExy[2]*qr_JetMET_MExy[3] > 0.5 )
    dc_JetMET[1]=1.;

//   std::cout << qr_JetMET_MExy[0] << " " 
// 	    << qr_JetMET_MExy[1] << " " 
// 	    << qr_JetMET_METPhi[0] << std::endl;
//   std::cout << qr_JetMET_MExy[2] << " " 
// 	    << qr_JetMET_MExy[3] << " " 
// 	    << qr_JetMET_METPhi[1] << std::endl;
//   std::cout << dc_JetMET[0] << " " << dc_JetMET[1] << std::endl;

  mMETDCFL1->Fill(dc_JetMET[0]*dc_JetMET[1]);
  mMETDCFL3[0]->Fill(dc_JetMET[0]);
  mMETDCFL3[1]->Fill(dc_JetMET[1]);

}

//define this as a plug-in
DEFINE_FWK_MODULE(DataCertificationJetMET);
