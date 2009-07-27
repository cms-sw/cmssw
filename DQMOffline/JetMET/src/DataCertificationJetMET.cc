// -*- C++ -*-
//
// Package:    DQMOffline/JetMET
// Class:      DataCertificationJetMET
// 
// Original Author:  "Frank Chlebana"
//         Created:  Sun Oct  5 13:57:25 CDT 2008
// $Id: DataCertificationJetMET.cc,v 1.27 2009/07/13 11:09:20 hatake Exp $
//

#include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"

// Some switches
#define NJetAlgo  4
#define NL3JFlags 3
#define NMETAlgo  3
#define NL3MFlags 3

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
  // verbose_ 0: suppress printouts
  //          1: show printouts
  verbose_ = conf_.getUntrackedParameter<int>("Verbose");

  if (verbose_) std::cout << ">>> BeginJob (DataCertificationJetMET) <<<" << std::endl;

  // -----------------------------------------
  //
  dbe_ = edm::Service<DQMStore>().operator->();

  // -----------------------------------------
  // testType 0: no comparison with histograms
  //          1: KS test
  //          2: Chi2 test
  //
  testType_ = 0; 
  testType_ = conf_.getUntrackedParameter<int>("TestType");
  if (verbose_) std::cout << ">>> TestType_        = " <<  testType_  << std::endl;  

  std::string filename    = conf_.getUntrackedParameter<std::string>("fileName");
  if (verbose_) std::cout << ">>> FileName        = " << filename    << std::endl;
  InMemory_ = true;

  //
  //--- If fileName is not defined, it means the the monitoring MEs are already in memory.
  if (filename != "") InMemory_ = false;
  if (verbose_) std::cout << "InMemory_           = " << InMemory_    << std::endl;

  //
  //--- If fileName is defined, read the test file and reference file and store in DQMStore.
  if (!InMemory_) {

    std::string filename    = conf_.getUntrackedParameter<std::string>("fileName");
    if (verbose_) std::cout << "FileName           = " << filename    << std::endl;

    std::string reffilename;
    if (testType_>=1){
      reffilename = conf_.getUntrackedParameter<std::string>("refFileName");
      if (verbose_) std::cout << "Reference FileName = " << reffilename << std::endl;
    }

    // -- Current & Reference Run
    //---------------------------------------------
    dbe_->load(filename);
    //dbe_->open(filename);
    //dbe_->open(filename,false,"","Collate");
    //dbe_->setCurrentFolder("/Collate");
    //if (testType_>=1) dbe_->open(reffilename);

  }

}

// ------------ method called once each job after finishing event loop  ------------
void 
DataCertificationJetMET::endJob()
{

  if (verbose_) std::cout << ">>> EndJob (DataCertificationJetMET) <<<" << std::endl;

  bool outputFile            = conf_.getUntrackedParameter<bool>("OutputFile");
  std::string outputFileName = conf_.getUntrackedParameter<std::string>("OutputFileName");
  if (verbose_) std::cout << ">>> endJob " << outputFile << std:: endl;

  if(outputFile){
    dbe_->showDirStructure();
    dbe_->save(outputFileName,
	       "", "","",
	       (DQMStore::SaveReferenceTag) DQMStore::SaveWithReference);
  }

}
 
// ------------ method called just before starting a new lumi section  ------------
void 
DataCertificationJetMET::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& c)
{

  if (verbose_) std::cout << ">>> BeginLuminosityBlock (DataCertificationJetMET) <<<" << std::endl;
  if (verbose_) std::cout << ">>> lumiBlock = " << lumiBlock.id()                   << std::endl;
  if (verbose_) std::cout << ">>> run       = " << lumiBlock.id().run()             << std::endl;
  if (verbose_) std::cout << ">>> lumiBlock = " << lumiBlock.id().luminosityBlock() << std::endl;

}

// ------------ method called just after a lumi section ends  ------------
void 
DataCertificationJetMET::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& c)
{

  if (verbose_) std::cout << ">>> EndLuminosityBlock (DataCertificationJetMET) <<<" << std::endl;
  if (verbose_) std::cout << ">>> lumiBlock = " << lumiBlock.id()                   << std::endl;
  if (verbose_) std::cout << ">>> run       = " << lumiBlock.id().run()             << std::endl;
  if (verbose_) std::cout << ">>> lumiBlock = " << lumiBlock.id().luminosityBlock() << std::endl;

  if (verbose_) dbe_->showDirStructure();  

  //
  //-----
  /*
  MonitorElement * meMETPhi=0;
  meMETPhi = new MonitorElement(*(dbe_->get("JetMET/MET/CaloMET/METTask_CaloMETPhi")));
  const QReport * myQReport = meMETPhi->getQReport("phiQTest"); //get QReport associated to your ME  
  if(myQReport) {
    float qtresult = myQReport->getQTresult(); // get QT result value
    int qtstatus   = myQReport->getStatus() ;  // get QT status value (see table below)
    std::string qtmessage = myQReport->getMessage() ; // get the whole QT result message
    if (verbose_) std::cout << "test" << qtmessage << " qtresult = " << qtresult << " qtstatus = " << qtstatus << std::endl;    
  }
  */

}

// ------------ method called just before starting a new run  ------------
void 
DataCertificationJetMET::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  if (verbose_) std::cout << ">>> BeginRun (DataCertificationJetMET) <<<" << std::endl;
  //if (verbose_) std::cout << ">>> run = " << run.id() << std::endl;

}

// ------------ method called right after a run ends ------------
void 
DataCertificationJetMET::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  
  if (verbose_) std::cout << ">>> EndRun (DataCertificationJetMET) <<<" << std::endl;
  //if (verbose_) std::cout << ">>> run = " << run.id() << std::endl;

  // -----------------------------------------

  std::vector<MonitorElement*> mes;
  std::vector<std::string> subDirVec;
  std::string RunDir;
  std::string RunNum;
  int         RunNumber=0;

  std::string RefRunDir;

  if (verbose_) std::cout << "InMemory_           = " << InMemory_    << std::endl;

  if (InMemory_) {
    //----------------------------------------------------------------
    // Histograms are in memory (for standard full-chain mode)
    //----------------------------------------------------------------

    mes = dbe_->getAllContents("");
    if (verbose_) std::cout << "1 >>> found " <<  mes.size() << " monitoring elements!" << std::endl;

    dbe_->setCurrentFolder("JetMET");
    subDirVec = dbe_->getSubdirs();
    for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {    
      if (verbose_) std::cout << "-AAA- Dir = >>" << ic->c_str() << "<<" << std::endl;
    }

    RunDir    = "";
    RunNumber = run.id().run();

  } else {
    //----------------------------------------------------------------
    // Open input files (for standalone mode)
    //----------------------------------------------------------------

    mes = dbe_->getAllContents("");
    if (verbose_) std::cout << "found " << mes.size() << " monitoring elements!" << std::endl;
    
    dbe_->setCurrentFolder("/");
    std::string currDir = dbe_->pwd();
    if (verbose_) std::cout << "--- Current Directory " << currDir << std::endl;

    subDirVec = dbe_->getSubdirs();

    // *** If the same file is read in then we have only one subdirectory
    int ind = 0;
    for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {
      RunDir = *ic;
      RunNum = *ic;
      if (verbose_) std::cout << "-XXX- Dir = >>" << ic->c_str() << "<<" << std::endl;
      ind++;
    }

    //
    // Current
    //
    if (RunDir == "JetMET") {
      RunDir = "";
      if (verbose_) std::cout << "-XXX- RunDir = >>" << RunDir.c_str() << "<<" << std::endl;
    }
    RunNum.erase(0,4);
    if (RunNum!="")
    RunNumber = atoi(RunNum.c_str());
    if (verbose_) std::cout << "--- >>" << RunNumber << "<<" << std::endl;

  }

  if (verbose_) dbe_->showDirStructure();

  //----------------------------------------------------------------
  // Book integers/histograms for data certification results
  //----------------------------------------------------------------

  std::string Jet_Tag_L2[NJetAlgo];
  Jet_Tag_L2[0] = "JetMET_Jet_ICone";
  Jet_Tag_L2[1] = "JetMET_Jet_SISCone";
  Jet_Tag_L2[2] = "JetMET_Jet_PFlow";
  Jet_Tag_L2[3] = "JetMET_Jet_JPT";

  std::string Jet_Tag_L3[NJetAlgo][NL3JFlags];
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

  std::string MET_Tag_L2[NMETAlgo];
  MET_Tag_L2[0] = "JetMET_MET_CaloMET";
  MET_Tag_L2[1] = "JetMET_MET_tcMET";
  MET_Tag_L2[2] = "JetMET_MET_pfMET";

  std::string MET_Tag_L3[NJetAlgo][NL3MFlags];
  MET_Tag_L3[0][0] = "JetMET_MET_CaloMET_NoHF";
  MET_Tag_L3[0][1] = "JetMET_MET_CaloMET_HO";
  MET_Tag_L3[0][2] = "JetMET_MET_CaloMET_NoHFHO";

  if (RunDir=="Reference") RunDir="";
  if (verbose_) std::cout << RunDir << std::endl;
  dbe_->setCurrentFolder("JetMET/EventInfo/Certification/");    

  //
  // Layer 1
  //---------
  MonitorElement* mJetDCFL1 = dbe_->bookFloat("JetMET_Jet");
  MonitorElement* mMETDCFL1 = dbe_->bookFloat("JetMET_MET");

  //
  // Layer 2
  //---------
  MonitorElement* mJetDCFL2[10];
  MonitorElement* mMETDCFL2[10];

  for (int itag=0; itag<NJetAlgo; itag++) mJetDCFL2[itag] = dbe_->bookFloat(Jet_Tag_L2[itag]);
  for (int itag=0; itag<NMETAlgo; itag++) mMETDCFL2[itag] = dbe_->bookFloat(MET_Tag_L2[itag]);

  //
  // Layer 3
  //---------
  MonitorElement* mJetDCFL3[20];
  MonitorElement* mMETDCFL3[20];

  int iL3JetTags=0;
  for (int ialg=0; ialg<NJetAlgo; ialg++){
    for (int idet=0; idet<3; idet++){
      mJetDCFL3[iL3JetTags++]= dbe_->bookFloat(Jet_Tag_L3[ialg][idet]);
    }
  }
  int iL3METTags=0;
  int ialg=0; // CaloMET only
  for (int idet=0; idet<3; idet++){
    mMETDCFL3[iL3METTags++]= dbe_->bookFloat(MET_Tag_L3[ialg][idet]);
  }

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
  Int_t Jet_DCF_L3[NJetAlgo][NL3JFlags];
  
  std::string refHistoName;
  std::string newHistoName;

  MonitorElement * meNew;
  //MonitorElement * meRef;

  // --- Loop over jet algorithms for Layer 2
  for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    

    if (iAlgo == 0) {
        refHistoName = "JetMET/Jet/IterativeConeJets/";
        newHistoName = "JetMET/Jet/IterativeConeJets/";
    }
    if (iAlgo == 1) {
        refHistoName = "JetMET/Jet/SISConeJets/";
        newHistoName = "JetMET/Jet/SISConeJets/";
    }
    if (iAlgo == 2) {
        refHistoName = "JetMET/Jet/PFJets/";
        newHistoName = "JetMET/Jet/PFJets/";
    }
    if (iAlgo == 3) {
        refHistoName = "JetMET/Jet/JPTJets/";
        newHistoName = "JetMET/Jet/JPTJets/";
    }

    // ----------------
    // --- Layer 2

    test_Pt           = 0.;
    test_Eta          = 0.;
    test_Phi          = 0.;
    test_Constituents = 0.;
    test_HFrac        = 0.;

    meNew = dbe_->get(newHistoName+"Pt");    
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	//std::cout << refHisto->GetEntries() << std::endl;
	//std::cout << newHisto->GetEntries() << std::endl;
	switch (testType_) {
	case 1 :
	  test_Pt = newHisto->KolmogorovTest(refHisto,"UO");
	  break;
	case 2 :
	  test_Pt = newHisto->Chi2Test(refHisto);
	  break;
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Pt << std::endl;    
      }
    }

    meNew = dbe_->get(newHistoName+"Eta");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Eta = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_Eta = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Eta << std::endl;    
      }
    }

    meNew = dbe_->get(newHistoName+"Phi");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Phi = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_Phi = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Phi << std::endl;    
      }
    }
     
    meNew = dbe_->get(newHistoName+"Constituents");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :	  
	  test_Constituents = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Constituents = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Constituents << std::endl;    
      }
    }
     
    meNew = dbe_->get(newHistoName+"HFrac");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_HFrac = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_HFrac = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_HFrac << std::endl;    	
      }
    }

    if (verbose_)
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

    meNew = dbe_->get(newHistoName+"Pt_Barrel");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Pt_Barrel = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Pt_Barrel = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Pt_Barrel << std::endl;    	
      }
    }

    meNew = dbe_->get(newHistoName+"Phi_Barrel");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Phi_Barrel = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Phi_Barrel = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Phi_Barrel << std::endl;    	
      }
    }

    // --- EndCap
    meNew = dbe_->get(newHistoName+"Pt_EndCap");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Pt_EndCap = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Pt_EndCap = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Pt_EndCap << std::endl;    	
      }
    }

    meNew = dbe_->get(newHistoName+"Phi_EndCap");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Phi_EndCap = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Phi_EndCap = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Phi_EndCap << std::endl;    	

      }
    }

    // --- Forward
    meNew = dbe_->get(newHistoName+"Pt_Forward");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Pt_Forward = newHisto->KolmogorovTest(refHisto,"UO");
	case 2:
	  test_Pt_Forward = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Pt_Forward << std::endl;    	
      }
    }

    meNew = dbe_->get(newHistoName+"Phi_Forward");
    if (meNew->getRootObject() && meNew->getRefRootObject()) {
      TH1F *refHisto = meNew->getRefTH1F();
      TH1F *newHisto = meNew->getTH1F();
      if ((refHisto) && (newHisto)) {
	switch (testType_) {
	case 1 :
	  test_Phi_Forward = newHisto->KolmogorovTest(refHisto,"UO");
	case 2 :
	  test_Phi_Forward = newHisto->Chi2Test(refHisto);
	}
	if (verbose_ > 0) std::cout << ">>> Test (" << testType_ 
				   << ") Result = " << test_Phi_Forward << std::endl;    	

      }
    }

    if (verbose_)
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
    mJetDCFL3[iAlgo*NL3JFlags+0]->Fill(double(Jet_DCF_L3[iAlgo][0]));

    if ( (test_Pt_EndCap > 0.95) && (test_Phi_EndCap > 0.95) ) {
      Jet_DCF_L3[iAlgo][1] = 1;
    } else {
      Jet_DCF_L3[iAlgo][1] = 0;
    }
    // --- Fill DC results histogram
    mJetDCFL3[iAlgo*NL3JFlags+1]->Fill(double(Jet_DCF_L3[iAlgo][1]));

    if ( (test_Pt_Forward > 0.95) && (test_Phi_Forward > 0.95) ) {
      Jet_DCF_L3[iAlgo][2] = 1;
    } else {
      Jet_DCF_L3[iAlgo][2] = 0;
    }
    // --- Fill DC results histogram
    mJetDCFL3[iAlgo*NL3JFlags+2]->Fill(double(Jet_DCF_L3[iAlgo][2]));

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
  if (verbose_) {
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
      for (int iL3Flag=0; iL3Flag<NL3JFlags; iL3Flag++) {    
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
  // Prepare test histograms
  //
  MonitorElement *meMExy[6];
  MonitorElement *meMETPhi[3];

  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/MET/";
  else              newHistoName = RunDir+"/JetMET/Run summary/MET/";

  meMExy[0]   = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/METTask_CaloMEx"))));
  meMExy[1]   = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/METTask_CaloMEy"))));
  meMExy[2]   = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/METTask_CaloMEx"))));
  meMExy[3]   = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/METTask_CaloMEy"))));
  meMETPhi[0] = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/METTask_CaloMETPhi"))));
  meMETPhi[1] = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/METTask_CaloMETPhi"))));
				   
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
    if (verbose_) std::cout << QReport_MExy[0]->getMessage() << std::endl;
  }

  if (QReport_MExy[1]){
    if (QReport_MExy[1]->getStatus()==100) 
      qr_JetMET_MExy[1] = QReport_MExy[1]->getQTresult();
    if (verbose_) std::cout << QReport_MExy[1]->getMessage() << std::endl;
  }

  if (QReport_MExy[2]){
    if (QReport_MExy[2]->getStatus()==100) 
      qr_JetMET_MExy[2] = QReport_MExy[2]->getQTresult();
    if (verbose_) std::cout << QReport_MExy[2]->getMessage() << std::endl;
  }

  if (QReport_MExy[3]){
    if (QReport_MExy[3]->getStatus()==100) 
      qr_JetMET_MExy[3] = QReport_MExy[3]->getQTresult();
    if (verbose_) std::cout << QReport_MExy[3]->getMessage() << std::endl;
  }

  if (QReport_METPhi[0]){
    if (QReport_METPhi[0]->getStatus()==100) 
      qr_JetMET_METPhi[0] = QReport_METPhi[0]->getQTresult();
    if (verbose_) std::cout << QReport_METPhi[0]->getMessage() << std::endl;
  }

  if (QReport_METPhi[1]){
    if (QReport_METPhi[1]->getStatus()==100) 
      qr_JetMET_METPhi[1] = QReport_METPhi[1]->getQTresult();
    if (verbose_) std::cout << QReport_METPhi[1]->getMessage() << std::endl;
  }

  dc_JetMET[0]=0.;
  if (qr_JetMET_MExy[0]*qr_JetMET_MExy[1] > 0.5 )
    dc_JetMET[0]=1.;

  dc_JetMET[1]=0.;
  if (qr_JetMET_MExy[2]*qr_JetMET_MExy[3] > 0.5 )
    dc_JetMET[1]=1.;

  mMETDCFL1->Fill(dc_JetMET[0]*dc_JetMET[1]);
  mMETDCFL2[0]->Fill(dc_JetMET[0]);
  mMETDCFL3[0]->Fill(dc_JetMET[1]);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(DataCertificationJetMET);
