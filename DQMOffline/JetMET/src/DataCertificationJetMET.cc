// -*- C++ -*-
//
// Package:    DQMOffline/JetMET
// Class:      DataCertificationJetMET
// 
// Original Author:  "Frank Chlebana"
//         Created:  Sun Oct  5 13:57:25 CDT 2008
// $Id: DataCertificationJetMET.cc,v 1.38 2010/03/10 08:00:10 hatake Exp $
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
DataCertificationJetMET::beginJob(void)
{

  // -----------------------------------------
  // verbose_ 0: suppress printouts
  //          1: show printouts
  verbose_ = conf_.getUntrackedParameter<int>("Verbose");

  if (verbose_) std::cout << ">>> BeginJob (DataCertificationJetMET) <<<" << std::endl;

  // -----------------------------------------
  //
  dbe_ = edm::Service<DQMStore>().operator->();

  ////QTest KS test thresholds
  //jet_ks_thresh  = conf_.getUntrackedParameter<double>("jet_ks_thresh");
  //met_ks_thresh  = conf_.getUntrackedParameter<double>("met_ks_thresh");
  //met_phi_thresh = conf_.getUntrackedParameter<double>("met_phi_thresh");

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

  //----------

  dbe_->setCurrentFolder("JetMET/EventInfo/");    
  MonitorElement*  reportSummary = dbe_->bookFloat("reportSummary");
  MonitorElement*  CertificationSummary = dbe_->bookFloat("CertificationSummary");

  MonitorElement*  reportSummaryMap = dbe_->book2D("reportSummaryMap","reportSummaryMap",3,0,3,5,0,5);
  MonitorElement*  CertificationSummaryMap = dbe_->book2D("CertificationSummaryMap","CertificationSummaryMap",3,0,3,5,0,5);
  reportSummaryMap->getTH2F()->SetStats(kFALSE);
  CertificationSummaryMap->getTH2F()->SetStats(kFALSE);
  reportSummaryMap->getTH2F()->SetOption("colz");
  CertificationSummaryMap->getTH2F()->SetOption("colz");

  reportSummaryMap->setBinLabel(1,"CaloTower");
  reportSummaryMap->setBinLabel(2,"MET");
  reportSummaryMap->setBinLabel(3,"Jet");

  CertificationSummaryMap->setBinLabel(1,"CaloTower");
  CertificationSummaryMap->setBinLabel(2,"MET");
  CertificationSummaryMap->setBinLabel(3,"Jet");

  // 3,4: CaloJet Barrel
  // 3,3: CaloJet Endcap
  // 3,2: CaloJet Forward
  // 3,1: JPT
  // 3,0: PFJet

  // 2,4: CaloMET
  // 2,3: CaloMETNoHF
  // 2,2: TcMET
  // 2,1: PFMET
  // 2,0: MuonCorrectedMET

  // 1,4: Barrel
  // 1,3: EndCap
  // 1,2: Forward

  reportSummary->Fill(1.);
  CertificationSummary->Fill(1.);

  //----------------------------------------------------------------
  // Book integers/histograms for data certification results
  //----------------------------------------------------------------

  std::string Jet_Tag_L2[NJetAlgo];
  Jet_Tag_L2[0] = "JetMET_Jet_AntiKt";
  Jet_Tag_L2[1] = "JetMET_Jet_ICone";
  Jet_Tag_L2[2] = "JetMET_Jet_PFlow";
  Jet_Tag_L2[3] = "JetMET_Jet_JPT";

  std::string Jet_Tag_L3[NJetAlgo][NL3JFlags];
  Jet_Tag_L3[1][0] = "JetMET_Jet_AntiKt_Barrel";
  Jet_Tag_L3[1][1] = "JetMET_Jet_AntiKt_EndCap";
  Jet_Tag_L3[1][2] = "JetMET_Jet_AntiKt_Forward";
  Jet_Tag_L3[1][0] = "JetMET_Jet_ICone_Barrel";
  Jet_Tag_L3[1][1] = "JetMET_Jet_ICone_EndCap";
  Jet_Tag_L3[1][2] = "JetMET_Jet_ICone_Forward";
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
  dbe_->setCurrentFolder("JetMET/EventInfo/CertificationSummaryContents/");    

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
  //const int nLSBins=500;

  //-- old code
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
  //for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    
  for (int iAlgo=0; iAlgo<3; iAlgo++) {    // KH removing JPT for now to avoid crashs

    if (iAlgo == 0) {
        refHistoName = "JetMET/Jet/AntiKtJets/";
        newHistoName = "JetMET/Jet/AntiKtJets/";
    }
    if (iAlgo == 1) {
        refHistoName = "JetMET/Jet/IterativeConeJets/";
        newHistoName = "JetMET/Jet/IterativeConeJets/";
    }
    if (iAlgo == 2) {
        refHistoName = "JetMET/Jet/PFJets/";
        newHistoName = "JetMET/Jet/PFJets/";
    }
    if (iAlgo == 3) {
        refHistoName = "JetMET/Jet/JPTJets/";
        newHistoName = "JetMET/Jet/JPTJets/";
    }

    // Check if this folder exists. If not, skip.
    if (!dbe_->containsAnyMonitorable(newHistoName)) continue;
    
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

    std::cout << "ccc" << std::endl;

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
  //--- New code
  //-----------------------------
  // Jet DQM Data Certification
  //-----------------------------
  MonitorElement *meJetPt[5];
  MonitorElement *meJetEta[5];
  MonitorElement *meJetPhi[5];
  MonitorElement *meJetEMFrac[4];
  MonitorElement *meJetConstituents[4];
  MonitorElement *meJetNTracks;
 
  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/Jet/";
  else              newHistoName = RunDir+"/JetMET/Run summary/Jet/";

  //Jet Phi histos
  meJetPhi[0] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Phi_Barrel"))));
  meJetPhi[1] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Phi_EndCap"))));
  meJetPhi[2] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Phi_Forward"))));
  meJetPhi[3] = new MonitorElement(*(dbe_->get((newHistoName+"PFJets/Phi"))));
  meJetPhi[4] = new MonitorElement(*(dbe_->get((newHistoName+"JPT/Phi"))));

  //Jet Eta histos
  meJetEta[0] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Eta"))));
  meJetEta[1] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Eta"))));
  meJetEta[2] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Eta"))));
  meJetEta[3] = new MonitorElement(*(dbe_->get((newHistoName+"PFJets/Eta"))));
  meJetEta[4] = new MonitorElement(*(dbe_->get((newHistoName+"JPT/Eta"))));

  //Jet Pt histos
  meJetPt[0]  = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Pt_Barrel"))));
  meJetPt[1]  = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Pt_EndCap"))));
  meJetPt[2]  = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Pt_Forward"))));
  meJetPt[3]  = new MonitorElement(*(dbe_->get((newHistoName+"PFJets/Pt2"))));
  meJetPt[4]  = new MonitorElement(*(dbe_->get((newHistoName+"JPT/Pt2"))));

  ////Jet Constituents histos
  meJetConstituents[0] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Constituents"))));
  meJetConstituents[1] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Constituents"))));
  meJetConstituents[2] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/Constituents"))));
  meJetConstituents[3] = new MonitorElement(*(dbe_->get((newHistoName+"PFJets/Constituents"))));
  //
  ////Jet EMFrac histos
  meJetEMFrac[0] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/EFrac"))));
  meJetEMFrac[1] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/EFrac"))));
  meJetEMFrac[2] = new MonitorElement(*(dbe_->get((newHistoName+"AntiKtJets/EFrac"))));
  meJetEMFrac[3] = new MonitorElement(*(dbe_->get((newHistoName+"PFJets/EFrac"))));

  //JPT specific histos
  meJetNTracks = new MonitorElement(*(dbe_->get((newHistoName+"JPT/nTracks"))));
				   
  //------------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for Jets
  //--- Tests for Calo Barrel, EndCap and Forward, as well as PF and JPT jets
  //--- For Calo and PF jets:
  //--- Look at mean of Constituents, EM Frac and Pt
  //--- Look at Kolmogorov result for Eta, Phi, and Pt
  //--- For JPT jets:
  //--- Look at mean of Pt, AllPionsTrackNHits?, nTracks, 
  //--- Look at Kolmogorov result for Eta, Phi, and Pt
  //------------------------------------------------------------------------------

  //5 types of Jets {AK5 Barrel, AK5 EndCap, AK5 Forward, PF, JPT}
  //--- Method 1
  //--Kolmogorov test
  const QReport * QReport_JetEta[5];
  const QReport * QReport_JetPhi[5];
  //--Mean and KS tests
  //for Calo and PF jets
  const QReport * QReport_JetConstituents[4][2];
  const QReport * QReport_JetEFrac[4][2];
  const QReport * QReport_JetPt[5][2];
  //for JPT jets
  const QReport * QReport_JetNTracks[2];


  float qr_Jet_NTracks[2]         = {-1.};
  float qr_Jet_Constituents[4][2] = {{-1.}};
  float qr_Jet_EFrac[4][2]        = {{-1.}};
  float qr_Jet_Eta[5]             = {-1.};
  float qr_Jet_Phi[5]             = {-1.};
  float qr_Jet_Pt[5][2]           = {{-1.}};
  float dc_Jet[5]                 = {-1.};

  for (int jtyp = 0; jtyp < 5; ++jtyp){
    if (verbose_) std::cout<<"Executing loop on jetalg: "<<jtyp<<std::endl;
    //Mean test results
    if (jtyp < 4){
      QReport_JetConstituents[jtyp][0] = meJetConstituents[jtyp]->getQReport("meanJetConstituentsTest"); //get QReport associated to Jet Constituents
      QReport_JetConstituents[jtyp][1] = meJetConstituents[jtyp]->getQReport("KolmogorovTest"); //get QReport associated to Jet Constituents
      QReport_JetEFrac[jtyp][0]     = meJetEMFrac[jtyp]->getQReport("meanEMFractionTest"); //get QReport associated to Jet EM fraction
      QReport_JetEFrac[jtyp][1]     = meJetEMFrac[jtyp]->getQReport("KolmogorovTest"); //get QReport associated to Jet EM fraction
    }
    else {
      QReport_JetNTracks[0]    = meJetNTracks->getQReport("meanNTracksTest"); //get QReport associated to JPTNTracks
      QReport_JetNTracks[1]    = meJetNTracks->getQReport("KolmogorovTest"); //get QReport associated to JPTNTracks
    }
    QReport_JetPt[jtyp][0] = meJetPt[jtyp]->getQReport("meanJetPtTest"); //get QReport associated to Jet Pt
    //Kolmogorov test results
    QReport_JetPt[jtyp][1] = meJetPt[jtyp]->getQReport("KolmogorovTest"); //get QReport associated to Jet Pt
    QReport_JetPhi[jtyp]   = meJetPhi[jtyp]->getQReport("KolmogorovTest"); //get QReport associated to Jet Phi
    QReport_JetEta[jtyp]   = meJetEta[jtyp]->getQReport("KolmogorovTest"); //get QReport associated to Jet Eta
    
    //Jet Pt test
    if (QReport_JetPt[jtyp][0]){
      if (QReport_JetPt[jtyp][0]->getStatus()==100 ||
	  QReport_JetPt[jtyp][0]->getStatus()==200)
	//qr_Jet_Pt[jtyp][0] = QReport_JetPt[jtyp][0]->getQTresult();
	qr_Jet_Pt[jtyp][0] = 1;
      else if (QReport_JetPt[jtyp][0]->getStatus()==300)
	qr_Jet_Pt[jtyp][0] = 0;
      else 
	qr_Jet_Pt[jtyp][0] = -1;

      if (verbose_) std::cout<<"Found JetPt test on mean"<<std::endl;
      if (verbose_) std::cout << QReport_JetPt[jtyp][0]->getMessage() << std::endl;
      if (verbose_) std::cout << QReport_JetPt[jtyp][0]->getStatus() << std::endl;
      if (verbose_) std::cout << QReport_JetPt[jtyp][0]->getQTresult() << std::endl;
    }
    else qr_Jet_Pt[jtyp][0] = -2;
    
    if (QReport_JetPt[jtyp][1]){
      if (QReport_JetPt[jtyp][1]->getStatus()==100 ||
	  QReport_JetPt[jtyp][1]->getStatus()==200) 
	//qr_Jet_Pt[jtyp][1] = QReport_JetPt[jtyp][1]->getQTresult();
	qr_Jet_Pt[jtyp][1] = 1;
      else if (QReport_JetPt[jtyp][1]->getStatus()==300) 
	qr_Jet_Pt[jtyp][1] = 0;
      else
	qr_Jet_Pt[jtyp][1] = 0;

      if (verbose_) std::cout<<"Found JetPt KS test"<<std::endl;
      if (verbose_) std::cout << QReport_JetPt[jtyp][1]->getMessage() << std::endl;
      if (verbose_) std::cout << QReport_JetPt[jtyp][1]->getStatus() << std::endl;
      if (verbose_) std::cout << QReport_JetPt[jtyp][1]->getQTresult() << std::endl;
    }
    else qr_Jet_Pt[jtyp][1] = -2;
    
    //Jet Phi test
    if (QReport_JetPhi[jtyp]){
      if (QReport_JetPhi[jtyp]->getStatus()==100 ||
	  QReport_JetPhi[jtyp]->getStatus()==200) 
	//qr_Jet_Phi[jtyp] = QReport_JetPhi[jtyp]->getQTresult();
	qr_Jet_Phi[jtyp] = 1;
      else if (QReport_JetPhi[jtyp]->getStatus()==300)
	qr_Jet_Phi[jtyp] = 0;
      else
	qr_Jet_Phi[jtyp] = -1;

      if (verbose_) std::cout<<"Found JetPhi KS test"<<std::endl;
      if (verbose_) std::cout << QReport_JetPhi[jtyp]->getMessage() << std::endl;
      if (verbose_) std::cout << QReport_JetPhi[jtyp]->getStatus() << std::endl;
      if (verbose_) std::cout << QReport_JetPhi[jtyp]->getQTresult() << std::endl;
    }
    else qr_Jet_Phi[jtyp] = -2;

    //Jet Eta test
    if (QReport_JetEta[jtyp]){
      if (QReport_JetEta[jtyp]->getStatus()==100 ||
	  QReport_JetEta[jtyp]->getStatus()==200) 
	//qr_Jet_Eta[jtyp] = QReport_JetEta[jtyp]->getQTresult();
	qr_Jet_Eta[jtyp] = 1;
      else if (QReport_JetEta[jtyp]->getStatus()==300) 
	qr_Jet_Eta[jtyp] = 0;
      else
	qr_Jet_Eta[jtyp] = -1;

      if (verbose_) std::cout<<"Found JetEta KS test"<<std::endl;
      if (verbose_) std::cout << QReport_JetEta[jtyp]->getMessage() << std::endl;
      if (verbose_) std::cout << QReport_JetEta[jtyp]->getStatus() << std::endl;
      if (verbose_) std::cout << QReport_JetEta[jtyp]->getQTresult() << std::endl;
    }
    else qr_Jet_Eta[jtyp] = -2;

    if (jtyp < 4) {
      //Jet Constituents test
      if (QReport_JetConstituents[jtyp][0]){
      	if (QReport_JetConstituents[jtyp][0]->getStatus()==100 ||
	    QReport_JetConstituents[jtyp][0]->getStatus()==200) 
      	  //qr_Jet_Constituents[jtyp][0] = QReport_JetConstituents[jtyp][0]->getQTresult();
      	  qr_Jet_Constituents[jtyp][0] = 1;
	else if (QReport_JetConstituents[jtyp][0]->getStatus()==300) 
	  qr_Jet_Constituents[jtyp][0] = 0;
	else
	  qr_Jet_Constituents[jtyp][0] = -1;

	if (verbose_) std::cout<<"Found JetConstituents mean test"<<std::endl;
      	if (verbose_) std::cout << QReport_JetConstituents[jtyp][0]->getMessage() << std::endl;
      	if (verbose_) std::cout << QReport_JetConstituents[jtyp][0]->getStatus() << std::endl;
      	if (verbose_) std::cout << QReport_JetConstituents[jtyp][0]->getQTresult() << std::endl;
      }
      else qr_Jet_Constituents[jtyp][0] = -2;

      if (QReport_JetConstituents[jtyp][1]){
      	if (QReport_JetConstituents[jtyp][1]->getStatus()==100 ||
	    QReport_JetConstituents[jtyp][1]->getStatus()==200) 
      	  //qr_Jet_Constituents[jtyp][1] = QReport_JetConstituents[jtyp][1]->getQTresult();
      	  qr_Jet_Constituents[jtyp][1] = 1;
	else if (QReport_JetConstituents[jtyp][1]->getStatus()==300) 
	  qr_Jet_Constituents[jtyp][1] = 0;
	else
	  qr_Jet_Constituents[jtyp][1] = -1;

	if (verbose_) std::cout<<"Found JetConstituents KS test"<<std::endl;
      	if (verbose_) std::cout << QReport_JetConstituents[jtyp][1]->getMessage() << std::endl;
      	if (verbose_) std::cout << QReport_JetConstituents[jtyp][1]->getStatus() << std::endl;
      	if (verbose_) std::cout << QReport_JetConstituents[jtyp][1]->getQTresult() << std::endl;
      }
      else qr_Jet_Constituents[jtyp][1] = -2;

      //Jet EMFrac test
      if (QReport_JetEFrac[jtyp][0]){
	if (QReport_JetEFrac[jtyp][0]->getStatus()==100 ||
	    QReport_JetEFrac[jtyp][0]->getStatus()==200) 
	  //qr_Jet_EFrac[jtyp][0] = QReport_JetEFrac[jtyp][0]->getQTresult();
	  qr_Jet_EFrac[jtyp][0] = 1;
	else if (QReport_JetEFrac[jtyp][0]->getStatus()==300) 
	  qr_Jet_EFrac[jtyp][0] = 0;
	else
	  qr_Jet_EFrac[jtyp][0] = -1;

	if (verbose_) std::cout<<"Found JetEFrac mean test"<<std::endl;
	if (verbose_) std::cout << QReport_JetEFrac[jtyp][0]->getMessage() << std::endl;
	if (verbose_) std::cout << QReport_JetEFrac[jtyp][0]->getStatus() << std::endl;
	if (verbose_) std::cout << QReport_JetEFrac[jtyp][0]->getQTresult() << std::endl;
      }
      else qr_Jet_EFrac[jtyp][0] = -2;
      
      if (QReport_JetEFrac[jtyp][1]){
	if (QReport_JetEFrac[jtyp][1]->getStatus()==100 ||
	    QReport_JetEFrac[jtyp][1]->getStatus()==200) 
	  //qr_Jet_EFrac[jtyp][1] = QReport_JetEFrac[jtyp][1]->getQTresult();
	  qr_Jet_EFrac[jtyp][1] = 1;
	else if (QReport_JetEFrac[jtyp][1]->getStatus()==300) 
	  qr_Jet_EFrac[jtyp][1] = 0;
	else
	  qr_Jet_EFrac[jtyp][1] = -1;

	if (verbose_) std::cout<<"Found JetEFrac KS test"<<std::endl;
	if (verbose_) std::cout << QReport_JetEFrac[jtyp][1]->getMessage() << std::endl;
	if (verbose_) std::cout << QReport_JetEFrac[jtyp][1]->getStatus() << std::endl;
	if (verbose_) std::cout << QReport_JetEFrac[jtyp][1]->getQTresult() << std::endl;
      }
      else qr_Jet_EFrac[jtyp][1] = -2;
    }
    else {
      for (int ii = 0; ii < 2; ++ii) {
	//Jet NTracks test
	if (QReport_JetNTracks[ii]){
	  if (QReport_JetNTracks[ii]->getStatus()==100 ||
	      QReport_JetNTracks[ii]->getStatus()==200) 
	    //qr_Jet_NTracks[ii] = QReport_JetNTracks[ii]->getQTresult();
	    qr_Jet_NTracks[ii] = 1;
	  else if (QReport_JetNTracks[ii]->getStatus()==300) 
	    qr_Jet_NTracks[ii] = 0;
	  else
	    qr_Jet_NTracks[ii] = -1;
	  
	  if (verbose_) std::cout<<"Found JPTJetNTracks test"<<std::endl;
	  if (verbose_) std::cout << QReport_JetNTracks[ii]->getMessage() << std::endl;
	  if (verbose_) std::cout << QReport_JetNTracks[ii]->getStatus() << std::endl;
	  if (verbose_) std::cout << QReport_JetNTracks[ii]->getQTresult() << std::endl;
	}
	else qr_Jet_NTracks[ii] = -2;
      }
    }
    
    if (verbose_) {
      printf("====================Jet Type %d QTest Report Summary========================\n",jtyp);
      printf("Eta:    Phi:   Pt 1:    2:    Const/Ntracks 1:    2:    EFrac/tracknhits 1:    2:\n");
      if (jtyp<4) {
	printf("%2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f\n", \
	       qr_Jet_Eta[jtyp],					\
	       qr_Jet_Phi[jtyp],					\
	       qr_Jet_Pt[jtyp][0],					\
	       qr_Jet_Pt[jtyp][1],					\
	       qr_Jet_Constituents[jtyp][0],				\
	       qr_Jet_Constituents[jtyp][1],				\
	       qr_Jet_EFrac[jtyp][0],					\
	       qr_Jet_EFrac[jtyp][1]);
      }
      else {
	printf("%2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f\n",	\
	       qr_Jet_Eta[jtyp],					\
	       qr_Jet_Phi[jtyp],					\
	       qr_Jet_Pt[jtyp][0],					\
	       qr_Jet_Pt[jtyp][1],					\
	       qr_Jet_NTracks[0],					\
	       qr_Jet_NTracks[1]);
      }
      printf("===========================================================================\n");
    }
    //certification result for Jet
    
    if (jtyp < 4) {
      if ( (qr_Jet_EFrac[jtyp][0]        == 0) ||
	   (qr_Jet_EFrac[jtyp][1]        == 0) ||
	   (qr_Jet_Constituents[jtyp][1] == 0) || 
	   (qr_Jet_Constituents[jtyp][0] == 0) ||
	   (qr_Jet_Eta[jtyp]             == 0) ||
	   (qr_Jet_Phi[jtyp]             == 0) ||
	   (qr_Jet_Pt[jtyp][0]           == 0) ||
	   (qr_Jet_Pt[jtyp][1]           == 0)
	   )
	dc_Jet[jtyp] = 0;
      else if ( (qr_Jet_EFrac[jtyp][0]        == -1) &&
		(qr_Jet_EFrac[jtyp][1]        == -1) &&
		(qr_Jet_Constituents[jtyp][1] == -1) && 
		(qr_Jet_Constituents[jtyp][0] == -1) &&
		(qr_Jet_Eta[jtyp]             == -1) &&
		(qr_Jet_Phi[jtyp]             == -1) &&
		(qr_Jet_Pt[jtyp][0]           == -1) &&
		(qr_Jet_Pt[jtyp][1]           == -1 )
		)
	dc_Jet[jtyp] = -1;
      else
	dc_Jet[jtyp] = 1;
    }
    else {
      if ( (qr_Jet_NTracks[0]  == 0) || 
	   (qr_Jet_NTracks[1]  == 0) ||
	   (qr_Jet_Eta[jtyp]   == 0) ||
	   (qr_Jet_Phi[jtyp]   == 0) ||
	   (qr_Jet_Pt[jtyp][0] == 0) ||
	   (qr_Jet_Pt[jtyp][1] == 0)
	   )
	dc_Jet[jtyp] = 0;
      else if ( (qr_Jet_NTracks[0]  == -1) && 
		(qr_Jet_NTracks[1]  == -1) &&
		(qr_Jet_Eta[jtyp]   == -1) &&
		(qr_Jet_Phi[jtyp]   == -1) &&
		(qr_Jet_Pt[jtyp][0] == -1) &&
		(qr_Jet_Pt[jtyp][1] == -1)
		)
	dc_Jet[jtyp] = -1;
      else
	dc_Jet[jtyp] = 1;
    }
    
    if (verbose_) std::cout<<"Certifying Jet algo: "<<jtyp<<" with value: "<<dc_Jet[jtyp]<<std::endl;
    CertificationSummaryMap->Fill(3., jtyp, dc_Jet[jtyp]);
    reportSummaryMap->Fill(3., jtyp, dc_Jet[jtyp]);
  }


  //-----------------------------
  // MET DQM Data Certification
  //-----------------------------

  //
  // Prepare test histograms
  //
  MonitorElement *meMExy[5][2];
  MonitorElement *meMEt[5];
  MonitorElement *meSumEt[5];
  MonitorElement *meMETPhi[5];
  //MonitorElement *meMETEMFrac[5];
  //MonitorElement *meMETEmEt[3][2];
  //MonitorElement *meMETHadEt[3][2];
 
  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/MET/";
  else              newHistoName = RunDir+"/JetMET/Run summary/MET/";

  //MEx/MEy monitor elements
  meMExy[0][0] = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/All/METTask_CaloMEx"))));
  meMExy[0][1] = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/All/METTask_CaloMEy"))));
  meMExy[1][0] = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/All/METTask_CaloMEx"))));
  meMExy[1][1] = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/All/METTask_CaloMEy"))));
  meMExy[2][0] = new MonitorElement(*(dbe_->get((newHistoName+"PfMET/All/METTask_PfMEx"))));
  meMExy[2][1] = new MonitorElement(*(dbe_->get((newHistoName+"PfMET/All/METTask_PfMEy"))));
  meMExy[3][0] = new MonitorElement(*(dbe_->get((newHistoName+"TcMET/All/METTask_MEx"))));
  meMExy[3][1] = new MonitorElement(*(dbe_->get((newHistoName+"TcMET/All/METTask_MEy"))));
  meMExy[4][0] = new MonitorElement(*(dbe_->get((newHistoName+"MuCorrMET/All/METTask_CaloMEx"))));
  meMExy[4][1] = new MonitorElement(*(dbe_->get((newHistoName+"MuCorrMET/All/METTask_CaloMEy"))));
  //MET Phi monitor elements
  meMETPhi[0]  = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/All/METTask_CaloMETPhi"))));
  meMETPhi[1]  = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/All/METTask_CaloMETPhi"))));
  meMETPhi[2]  = new MonitorElement(*(dbe_->get((newHistoName+"PfMET/All/METTask_PfMETPhi"))));
  meMETPhi[3]  = new MonitorElement(*(dbe_->get((newHistoName+"TcMET/All/METTask_METPhi"))));
  meMETPhi[4]  = new MonitorElement(*(dbe_->get((newHistoName+"MuCorrMET/All/METTask_CaloMETPhi"))));
  //MET monitor elements
  meMEt[0]  = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/All/METTask_CaloMET"))));
  meMEt[1]  = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/All/METTask_CaloMET"))));
  meMEt[2]  = new MonitorElement(*(dbe_->get((newHistoName+"PfMET/All/METTask_PfMET"))));
  meMEt[3]  = new MonitorElement(*(dbe_->get((newHistoName+"TcMET/All/METTask_MET"))));
  meMEt[4]  = new MonitorElement(*(dbe_->get((newHistoName+"MuCorrMET/All/METTask_CaloMET"))));
  //SumET monitor elements
  meSumEt[0]  = new MonitorElement(*(dbe_->get((newHistoName+"CaloMET/All/METTask_CaloSumET"))));
  meSumEt[1]  = new MonitorElement(*(dbe_->get((newHistoName+"CaloMETNoHF/All/METTask_CaloSumET"))));
  meSumEt[2]  = new MonitorElement(*(dbe_->get((newHistoName+"PfMET/All/METTask_PfSumET"))));
  meSumEt[3]  = new MonitorElement(*(dbe_->get((newHistoName+"TcMET/All/METTask_SumET"))));
  meSumEt[4]  = new MonitorElement(*(dbe_->get((newHistoName+"MuCorrMET/All/METTask_CaloSumET"))));
				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------

  //5 types of MET {CaloMET, CaloMETNoHF, PfMET, TcMET, MuCorrMET}
  //2 types of tests Mean test/Kolmogorov test
  const QReport * QReport_MExy[5][2][2];
  const QReport * QReport_MEt[5][2];
  const QReport * QReport_SumEt[5][2];
  //2 types of tests phiQTest and Kolmogorov test
  const QReport * QReport_METPhi[5][2];


  float qr_MET_MExy[5][2][2] = {{{-999.}}};
  float qr_MET_MEt[5][2]     = {{-999.}};
  float qr_MET_SumEt[5][2]   = {{-999.}};
  float qr_MET_METPhi[5][2]  = {{-999.}};
  float dc_MET[5]            = {-999.};

  for (int mtyp = 0; mtyp < 5; ++mtyp){
    if (verbose_) std::cout<<"Executing loop on metalg: "<<mtyp<<std::endl;
    //Mean test results
    QReport_MExy[mtyp][0][0] = meMExy[mtyp][0]->getQReport("meanMExyTest"); //get QReport associated to MEx  
    QReport_MExy[mtyp][0][1] = meMExy[mtyp][1]->getQReport("meanMExyTest"); //get QReport associated to MEy  
    QReport_MEt[mtyp][0]     = meMEt[mtyp]->getQReport("meanMETTTest"); //get QReport associated to MET
    QReport_SumEt[mtyp][0]   = meSumEt[mtyp]->getQReport("meanSumETTest"); //get QReport associated to SumET
    //phiQTest results
    QReport_METPhi[mtyp][0]  = meMETPhi[mtyp]->getQReport("phiQTest"); //get QReport associated to METPhi  
    //Kolmogorov test results
    QReport_METPhi[mtyp][1]  = meMETPhi[mtyp]->getQReport("KolmogorovTest"); //get QReport associated to METPhi  
    QReport_MExy[mtyp][1][0] = meMExy[mtyp][0]->getQReport("KolmogorovTest"); //get QReport associated to MEx  
    QReport_MExy[mtyp][1][1] = meMExy[mtyp][1]->getQReport("KolmogorovTest"); //get QReport associated to MEy  
    QReport_MEt[mtyp][1]     = meMEt[mtyp]->getQReport("KolmogorovTest"); //get QReport associated to MET
    QReport_SumEt[mtyp][1]   = meSumEt[mtyp]->getQReport("KolmogorovTest"); //get QReport associated to SumET
    
    for (int testtyp = 0; testtyp < 2; ++testtyp) {
      //MEx test
      if (QReport_MExy[mtyp][testtyp][0]){
	if (QReport_MExy[mtyp][testtyp][0]->getStatus()==100 ||
	    QReport_MExy[mtyp][testtyp][0]->getStatus()==200) 
	  //qr_MET_MExy[mtyp][testtyp][0] = QReport_MExy[mtyp][testtyp][0]->getQTresult();
	  qr_MET_MExy[mtyp][testtyp][0] = 1;
	else if (QReport_MExy[mtyp][testtyp][0]->getStatus()==300) 
	  //qr_MET_MExy[mtyp][testtyp][0] = QReport_MExy[mtyp][testtyp][0]->getQTresult();
	  qr_MET_MExy[mtyp][testtyp][0] = 0;
	else
	  //qr_MET_MExy[mtyp][testtyp][0] = QReport_MExy[mtyp][testtyp][0]->getQTresult();
	  qr_MET_MExy[mtyp][testtyp][0] = -1;

	if (verbose_) std::cout<<"Found MEx test "<<testtyp<<std::endl;
	if (verbose_) std::cout << QReport_MExy[mtyp][testtyp][0]->getMessage() << std::endl;
	if (verbose_) std::cout << QReport_MExy[mtyp][testtyp][0]->getStatus() << std::endl;
	if (verbose_) std::cout << QReport_MExy[mtyp][testtyp][0]->getQTresult() << std::endl;
      }
      else qr_MET_MExy[mtyp][testtyp][0] = -2;

      //MEy test
      if (QReport_MExy[mtyp][testtyp][1]){
	if (QReport_MExy[mtyp][testtyp][1]->getStatus()==100 ||
	    QReport_MExy[mtyp][testtyp][1]->getStatus()==200) 
	  //qr_MET_MExy[mtyp][testtyp][1] = QReport_MExy[mtyp][testtyp][1]->getQTresult();
	  qr_MET_MExy[mtyp][testtyp][1] = 1;
	else if (QReport_MExy[mtyp][testtyp][1]->getStatus()==300) 
	  //qr_MET_MExy[mtyp][testtyp][1] = QReport_MExy[mtyp][testtyp][1]->getQTresult();
	  qr_MET_MExy[mtyp][testtyp][1] = 0;
	else
	  //qr_MET_MExy[mtyp][testtyp][1] = QReport_MExy[mtyp][testtyp][1]->getQTresult();
	  qr_MET_MExy[mtyp][testtyp][1] = -1;

	if (verbose_) std::cout<<"Found MEy test "<<testtyp<<std::endl;
	if (verbose_) std::cout << QReport_MExy[mtyp][testtyp][1]->getMessage() << std::endl;
	if (verbose_) std::cout << QReport_MExy[mtyp][testtyp][1]->getStatus() << std::endl;
	if (verbose_) std::cout << QReport_MExy[mtyp][testtyp][1]->getQTresult() << std::endl;
      }
      else qr_MET_MExy[mtyp][testtyp][1] = -2;

      //MEt test
      if (QReport_MEt[mtyp][testtyp]){
	if (QReport_MEt[mtyp][testtyp]->getStatus()==100 ||
	    QReport_MEt[mtyp][testtyp]->getStatus()==200) 
	  //qr_MET_MEt[mtyp][testtyp] = QReport_MEt[mtyp][testtyp]->getQTresult();
	  qr_MET_MEt[mtyp][testtyp] = 1;
	else if (QReport_MEt[mtyp][testtyp]->getStatus()==300) 
	  //qr_MET_MEt[mtyp][testtyp] = QReport_MEt[mtyp][testtyp]->getQTresult();
	  qr_MET_MEt[mtyp][testtyp] = 0;
	else
	  //qr_MET_MEt[mtyp][testtyp] = QReport_MEt[mtyp][testtyp]->getQTresult();
	  qr_MET_MEt[mtyp][testtyp] = -1;

	if (verbose_) std::cout<<"Found MEt test "<<testtyp<<std::endl;
	if (verbose_) std::cout << QReport_MEt[mtyp][testtyp]->getMessage() << std::endl;
	if (verbose_) std::cout << QReport_MEt[mtyp][testtyp]->getStatus() << std::endl;
	if (verbose_) std::cout << QReport_MEt[mtyp][testtyp]->getQTresult() << std::endl;
      }
      else qr_MET_MEt[mtyp][testtyp] = -2;

      //SumEt test
      if (QReport_SumEt[mtyp][testtyp]){
	if (QReport_SumEt[mtyp][testtyp]->getStatus()==100 ||
	    QReport_SumEt[mtyp][testtyp]->getStatus()==200) 
	  //qr_MET_SumEt[mtyp][testtyp] = QReport_SumEt[mtyp][testtyp]->getQTresult();
	  qr_MET_SumEt[mtyp][testtyp] = 1;
	else if (QReport_SumEt[mtyp][testtyp]->getStatus()==300) 
	  //qr_MET_SumEt[mtyp][testtyp] = QReport_SumEt[mtyp][testtyp]->getQTresult();
	  qr_MET_SumEt[mtyp][testtyp] = 0;
	else
	  //qr_MET_SumEt[mtyp][testtyp] = QReport_SumEt[mtyp][testtyp]->getQTresult();
	  qr_MET_SumEt[mtyp][testtyp] = -1;

	if (verbose_) std::cout<<"Found SumEt test "<<testtyp<<std::endl;
	if (verbose_) std::cout << QReport_SumEt[mtyp][testtyp]->getMessage() << std::setw(5);
	if (verbose_) std::cout << QReport_SumEt[mtyp][testtyp]->getStatus() << std::setw(5);
	if (verbose_) std::cout << QReport_SumEt[mtyp][testtyp]->getQTresult() << std::endl;
      }
      else qr_MET_SumEt[mtyp][testtyp] = -2;

      //METPhi test
      if (QReport_METPhi[mtyp][testtyp]){
	if (verbose_) std::cout<<"Found METPhi test "<<testtyp<<std::endl;
	if (QReport_METPhi[mtyp][testtyp]->getStatus()==100 ||
	    QReport_METPhi[mtyp][testtyp]->getStatus()==200) 
	  //qr_MET_METPhi[mtyp][testtyp] = QReport_METPhi[mtyp][testtyp]->getQTresult();
	  qr_MET_METPhi[mtyp][testtyp] = 1;
	else if (QReport_METPhi[mtyp][testtyp]->getStatus()==300) 
	  //qr_MET_METPhi[mtyp][testtyp] = QReport_METPhi[mtyp][testtyp]->getQTresult();
	  qr_MET_METPhi[mtyp][testtyp] = 0;
	else
	  //qr_MET_METPhi[mtyp][testtyp] = QReport_METPhi[mtyp][testtyp]->getQTresult();
	  qr_MET_METPhi[mtyp][testtyp] = -1;
	if (verbose_) std::cout << QReport_METPhi[mtyp][testtyp]->getMessage() << std::endl;
	if (verbose_) std::cout << QReport_METPhi[mtyp][testtyp]->getStatus() << std::endl;
	if (verbose_) std::cout << QReport_METPhi[mtyp][testtyp]->getQTresult() << std::endl;
      }
      else qr_MET_METPhi[mtyp][testtyp] = -2;
    }

    if (verbose_) {
      //certification result for MET
      printf("====================MET Type %d QTest Report Summary========================\n",mtyp);
      printf("MEx test    MEy test    MEt test:    SumEt test:    METPhi test:\n");
      for (int tt = 0; tt < 2; ++tt) {
	printf("%2.2f    %2.2f    %2.2f    %2.2f    %2.2f\n",qr_MET_MExy[mtyp][tt][0], \
	       qr_MET_MExy[mtyp][tt][1],				\
	       qr_MET_MEt[mtyp][tt],					\
	       qr_MET_SumEt[mtyp][tt],					\
	       qr_MET_METPhi[mtyp][tt]);
      }
      printf("===========================================================================\n");
    }
    if ( 
	(qr_MET_MExy[mtyp][0][0] == 0) ||
	(qr_MET_MExy[mtyp][0][1] == 0) ||
	(qr_MET_MEt[mtyp][0]     == 0) ||
	(qr_MET_SumEt[mtyp][0]   == 0) ||
	(qr_MET_METPhi[mtyp][0]  == 0) ||
	(qr_MET_MExy[mtyp][1][0] == 0) ||
	(qr_MET_MExy[mtyp][1][1] == 0) ||
	(qr_MET_MEt[mtyp][1]     == 0) ||
	(qr_MET_SumEt[mtyp][1]   == 0) ||
	(qr_MET_METPhi[mtyp][1]  == 0)
	)
      dc_MET[mtyp] = 0;
    else if (
	     (qr_MET_MExy[mtyp][0][0] == -1) &&
	     (qr_MET_MExy[mtyp][0][1] == -1) &&
	     (qr_MET_MEt[mtyp][0]     == -1) &&
	     (qr_MET_SumEt[mtyp][0]   == -1) &&
	     (qr_MET_METPhi[mtyp][0]  == -1) &&
	     (qr_MET_MExy[mtyp][1][0] == -1) &&
	     (qr_MET_MExy[mtyp][1][1] == -1) &&
	     (qr_MET_MEt[mtyp][1]     == -1) &&
	     (qr_MET_SumEt[mtyp][1]   == -1) &&
	     (qr_MET_METPhi[mtyp][1]  == -1)
	     )
      dc_MET[mtyp] = -1;
    else
      dc_MET[mtyp] = 1;

    if (verbose_) std::cout<<"Certifying MET algo: "<<mtyp<<" with value: "<<dc_MET[mtyp]<<std::endl;
    CertificationSummaryMap->Fill(2., mtyp, dc_MET[mtyp]);
    reportSummaryMap->Fill(2., mtyp, dc_MET[mtyp]);
  }

  mMETDCFL1->Fill(dc_MET[0]*dc_MET[1]);
  mMETDCFL2[0]->Fill(dc_MET[0]);
  mMETDCFL3[0]->Fill(dc_MET[1]);


  dbe_->setCurrentFolder("");  

}

//define this as a plug-in
//DEFINE_FWK_MODULE(DataCertificationJetMET);
