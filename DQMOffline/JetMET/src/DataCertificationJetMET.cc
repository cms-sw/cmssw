// -*- C++ -*-
//
// Package:    DQMOffline/JetMET
// Class:      DataCertificationJetMET
// 
// Original Author:  "Frank Chlebana"
//         Created:  Sun Oct  5 13:57:25 CDT 2008
//

#include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

// Some switches
//
// constructors and destructor
//
DataCertificationJetMET::DataCertificationJetMET(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  // now do what ever initialization is needed

  // -----------------------------------------
  // verbose_ 0: suppress printouts
  //          1: show printouts
  verbose_ = conf_.getUntrackedParameter<int>("Verbose",0);
  metFolder   = conf_.getUntrackedParameter<std::string>("metFolder");
  jetAlgo     = conf_.getUntrackedParameter<std::string>("jetAlgo");
  folderName  = conf_.getUntrackedParameter<std::string>("folderName");

  jetTests[0][0] = conf_.getUntrackedParameter<bool>("pfBarrelJetMeanTest",true);
  jetTests[0][1] = conf_.getUntrackedParameter<bool>("pfBarrelJetKSTest",false);
  jetTests[1][0] = conf_.getUntrackedParameter<bool>("pfEndcapJetMeanTest",true);
  jetTests[1][1] = conf_.getUntrackedParameter<bool>("pfEndcapJetKSTest",false);
  jetTests[2][0] = conf_.getUntrackedParameter<bool>("pfForwardJetMeanTest",true);
  jetTests[2][1] = conf_.getUntrackedParameter<bool>("pfForwardJetKSTest",false);
  jetTests[3][0] = conf_.getUntrackedParameter<bool>("caloJetMeanTest",true);
  jetTests[3][1] = conf_.getUntrackedParameter<bool>("caloJetKSTest",false);
  jetTests[4][0] = conf_.getUntrackedParameter<bool>("jptJetMeanTest",true);
  jetTests[4][1] = conf_.getUntrackedParameter<bool>("jptJetKSTest",false);

  metTests[0][0] = conf_.getUntrackedParameter<bool>("caloMETMeanTest",true);
  metTests[0][1] = conf_.getUntrackedParameter<bool>("caloMETKSTest",false);
  metTests[1][0] = conf_.getUntrackedParameter<bool>("pfMETMeanTest",true);
  metTests[1][1] = conf_.getUntrackedParameter<bool>("pfMETKSTest",false);
  metTests[2][0] = conf_.getUntrackedParameter<bool>("tcMETMeanTest",true);
  metTests[2][1] = conf_.getUntrackedParameter<bool>("tcMETKSTest",false);
  dbe_ = edm::Service<DQMStore>().operator->();
 
  if (verbose_) std::cout << ">>> Constructor (DataCertificationJetMET) <<<" << std::endl;

  // -----------------------------------------
  //
}


DataCertificationJetMET::~DataCertificationJetMET()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (verbose_) std::cout << ">>> Deconstructor (DataCertificationJetMET) <<<" << std::endl;

  bool outputFile            = conf_.getUntrackedParameter<bool>("OutputFile");
  std::string outputFileName = conf_.getUntrackedParameter<std::string>("OutputFileName");
  if (verbose_) std::cout << ">>> deconstructor" << outputFile << std:: endl;

  if(outputFile){
    dbe_->showDirStructure();
    dbe_->save(outputFileName,
	       "", "","",
	       (DQMStore::SaveReferenceTag) DQMStore::SaveWithReference);
  }
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DataCertificationJetMET::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  isData = iEvent.isRealData();
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
DataCertificationJetMET::beginJob(void)
{

}

// ------------ method called once each job after finishing event loop  ------------
void 
DataCertificationJetMET::endJob()
{
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

void DataCertificationJetMET::bookHistograms(DQMStore::IBooker & ibooker,
				 edm::Run const & iRun,
				 edm::EventSetup const & ) {
  ibooker.setCurrentFolder(folderName);

  reportSummary = ibooker.bookFloat("reportSummary");
  CertificationSummary = ibooker.bookFloat("CertificationSummary");
  
  reportSummaryMap = ibooker.book2D("reportSummaryMap","reportSummaryMap",3,0,3,5,0,5);
  CertificationSummaryMap = ibooker.book2D("CertificationSummaryMap","CertificationSummaryMap",3,0,3,5,0,5);
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


  dbe_->setCurrentFolder(folderName);  
  
  reportSummary = dbe_->get(folderName+"/"+"reportSummary");
  CertificationSummary = dbe_->get(folderName+"/"+"CertificationSummary");
  reportSummaryMap = dbe_->get(folderName+"/"+"reportSummaryMap");
  CertificationSummaryMap = dbe_->get(folderName+"/"+"CertificationSummaryMap");


  
  if(reportSummaryMap && reportSummaryMap->getRootObject()){ 
    reportSummaryMap->getTH2F()->SetStats(kFALSE);
    reportSummaryMap->getTH2F()->SetOption("colz");
    reportSummaryMap->setBinLabel(1,"CaloTower");
    reportSummaryMap->setBinLabel(2,"MET");
    reportSummaryMap->setBinLabel(3,"Jet");
  }
  if(CertificationSummaryMap && CertificationSummaryMap->getRootObject()){ 
    CertificationSummaryMap->getTH2F()->SetStats(kFALSE);
    CertificationSummaryMap->getTH2F()->SetOption("colz");
    CertificationSummaryMap->setBinLabel(1,"CaloTower");
    CertificationSummaryMap->setBinLabel(2,"MET");
    CertificationSummaryMap->setBinLabel(3,"Jet");
  }

  reportSummary->Fill(1.);
  CertificationSummary->Fill(1.);

  if (RunDir=="Reference") RunDir="";
  if (verbose_) std::cout << RunDir << std::endl;
  dbe_->setCurrentFolder("JetMET/EventInfo/CertificationSummaryContents/");    


  std::string refHistoName;
  std::string newHistoName;
  
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
  else              newHistoName = RunDir+"/JetMET/Runsummary/Jet/";
  std::string cleaningdir = "";
  if (isData){ //directory should be present in MC as well -> take out a later stage maybe
    cleaningdir = "Cleaned";
  }else{
    cleaningdir = "Uncleaned";
  }
  //Jet Phi histos
  meJetPhi[0] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_Barrel");
  meJetPhi[1] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_EndCap");
  meJetPhi[2] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_Forward");
  meJetPhi[3] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Phi");
  meJetPhi[4] = dbe_->get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Phi");

  //Jet Eta histos
  meJetEta[0] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Eta");
  meJetEta[1] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Eta");
  meJetEta[2] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/EtaFirst");
  meJetEta[3] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Eta");
  meJetEta[4] = dbe_->get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Eta");

  //Jet Pt histos
  meJetPt[0]  = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_Barrel");
  meJetPt[1]  = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_EndCap");
  meJetPt[2]  = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_Forward");
  meJetPt[3]  = dbe_->get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Pt_2");
  meJetPt[4]  = dbe_->get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Pt_2");

  ////Jet Constituents histos
  meJetConstituents[0] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Barrel");
  meJetConstituents[1] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_EndCap");
  meJetConstituents[2] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Forward");
  meJetConstituents[3] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Constituents");
  //
  ////Jet EMFrac histos
  meJetEMFrac[0] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Barrel");
  meJetEMFrac[1] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_EndCap");
  meJetEMFrac[2] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Forward");
  meJetEMFrac[3] = dbe_->get(newHistoName+cleaningdir+jetAlgo+"CaloJets/EFrac");

  //JPT specific histos
  meJetNTracks = dbe_->get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/nTracks");
				   
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


  // Five types of jets {AK5 Barrel, AK5 EndCap, AK5 Forward, PF, JPT}
  //----------------------------------------------------------------------------
  // Kolmogorov (KS) tests
  const QReport* QReport_JetEta[5] = {0, 0, 0, 0, 0};
  const QReport* QReport_JetPhi[5] = {0, 0, 0, 0, 0};

  // Mean and KS tests for Calo and PF jets
  const QReport* QReport_JetConstituents[4][2] = {{0,0}, {0,0}, {0,0}, {0,0}};
  const QReport* QReport_JetEFrac[4][2]        = {{0,0}, {0,0}, {0,0}, {0,0}};
  const QReport* QReport_JetPt[5][2]           = {{0,0}, {0,0}, {0,0}, {0,0}, {0,0}};

  // Mean and KS tests for JPT jets
  const QReport* QReport_JetNTracks[2] = {0, 0};

  float qr_Jet_NTracks[2] = {-1, -1};
  float qr_Jet_Eta[5]     = {-1, -1, -1, -1, -1};
  float qr_Jet_Phi[5]     = {-1, -1, -1, -1, -1};
  float dc_Jet[5]         = {-1, -1, -1, -1, -1};

  float qr_Jet_Constituents[4][2] = {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}};
  float qr_Jet_EFrac[4][2]        = {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}};
  float qr_Jet_Pt[5][2]           = {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}};


  // Loop
  //----------------------------------------------------------------------------
  for (int jtyp=0; jtyp<5; ++jtyp) {
    // Mean test results
    if (jtyp < 4){
      if (meJetConstituents[jtyp] && meJetConstituents[jtyp]->getRootObject() ) {
	QReport_JetConstituents[jtyp][0] = meJetConstituents[jtyp]->getQReport("meanJetConstituentsTest");
	QReport_JetConstituents[jtyp][1] = meJetConstituents[jtyp]->getQReport("KolmogorovTest");
      }
      if (meJetEMFrac[jtyp]&& meJetEMFrac[jtyp]->getRootObject() ) {
	QReport_JetEFrac[jtyp][0]        = meJetEMFrac[jtyp]->getQReport("meanEMFractionTest");
	QReport_JetEFrac[jtyp][1]        = meJetEMFrac[jtyp]->getQReport("KolmogorovTest");
      }
    }
    else {
      if (meJetNTracks  && meJetNTracks->getRootObject() ) {
	QReport_JetNTracks[0]    = meJetNTracks->getQReport("meanNTracksTest");
	QReport_JetNTracks[1]    = meJetNTracks->getQReport("KolmogorovTest");
      }
    }
    if (meJetPt[jtyp] && meJetPt[jtyp]->getRootObject() ) {
      QReport_JetPt[jtyp][0] = meJetPt[jtyp]->getQReport("meanJetPtTest");
      QReport_JetPt[jtyp][1] = meJetPt[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetPhi[jtyp] && meJetPhi[jtyp]->getRootObject()){
      QReport_JetPhi[jtyp]   = meJetPhi[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetEta[jtyp] && meJetEta[jtyp]->getRootObject()){
      QReport_JetEta[jtyp]   = meJetEta[jtyp]->getQReport("KolmogorovTest");
    }
    
    //Jet Pt test
    if (QReport_JetPt[jtyp][0]){
      //std::cout<<"jet type test pt "<<jtyp<<"/"<<QReport_JetPt[jtyp][0]->getStatus()<<std::endl;
      if (QReport_JetPt[jtyp][0]->getStatus()==100 ||
	  QReport_JetPt[jtyp][0]->getStatus()==200)
	qr_Jet_Pt[jtyp][0] = 1;
      else if (QReport_JetPt[jtyp][0]->getStatus()==300)
	qr_Jet_Pt[jtyp][0] = 0;
      else 
	qr_Jet_Pt[jtyp][0] = -1;
    }
    else{ qr_Jet_Pt[jtyp][0] = -2;
      //std::cout<<"qreport is REALLY NULL type test pt "<<jtyp<<" 0 "<<std::endl;
    }
    if (QReport_JetPt[jtyp][1]){
      if (QReport_JetPt[jtyp][1]->getStatus()==100 ||
	  QReport_JetPt[jtyp][1]->getStatus()==200) 
	qr_Jet_Pt[jtyp][1] = 1;
      else if (QReport_JetPt[jtyp][1]->getStatus()==300) 
	qr_Jet_Pt[jtyp][1] = 0;
      else
	qr_Jet_Pt[jtyp][1] = -1;
    }
    else{ qr_Jet_Pt[jtyp][1] = -2;
    }
    
    //Jet Phi test
    if (QReport_JetPhi[jtyp]){
      if (QReport_JetPhi[jtyp]->getStatus()==100 ||
	  QReport_JetPhi[jtyp]->getStatus()==200) 
	qr_Jet_Phi[jtyp] = 1;
      else if (QReport_JetPhi[jtyp]->getStatus()==300)
	qr_Jet_Phi[jtyp] = 0;
      else
	qr_Jet_Phi[jtyp] = -1;
    }
    else{ qr_Jet_Phi[jtyp] = -2;
    }
    //Jet Eta test
    if (QReport_JetEta[jtyp]){
      if (QReport_JetEta[jtyp]->getStatus()==100 ||
	  QReport_JetEta[jtyp]->getStatus()==200) 
	qr_Jet_Eta[jtyp] = 1;
      else if (QReport_JetEta[jtyp]->getStatus()==300) 
	qr_Jet_Eta[jtyp] = 0;
      else
	qr_Jet_Eta[jtyp] = -1;
    }
    else{ 
      qr_Jet_Eta[jtyp] = -2;
    }
    if (jtyp < 4) {
      //Jet Constituents test
      if (QReport_JetConstituents[jtyp][0]){
      	if (QReport_JetConstituents[jtyp][0]->getStatus()==100 ||
	    QReport_JetConstituents[jtyp][0]->getStatus()==200) 
      	  qr_Jet_Constituents[jtyp][0] = 1;
	else if (QReport_JetConstituents[jtyp][0]->getStatus()==300) 
	  qr_Jet_Constituents[jtyp][0] = 0;
	else
	  qr_Jet_Constituents[jtyp][0] = -1;
      }
      else{ qr_Jet_Constituents[jtyp][0] = -2;
      }

      if (QReport_JetConstituents[jtyp][1]){
      	if (QReport_JetConstituents[jtyp][1]->getStatus()==100 ||
	    QReport_JetConstituents[jtyp][1]->getStatus()==200) 
      	  qr_Jet_Constituents[jtyp][1] = 1;
	else if (QReport_JetConstituents[jtyp][1]->getStatus()==300) 
	  qr_Jet_Constituents[jtyp][1] = 0;
	else
	  qr_Jet_Constituents[jtyp][1] = -1;
      }
      else{ qr_Jet_Constituents[jtyp][1] = -2;
      }
      //Jet EMFrac test
      if (QReport_JetEFrac[jtyp][0]){
	if (QReport_JetEFrac[jtyp][0]->getStatus()==100 ||
	    QReport_JetEFrac[jtyp][0]->getStatus()==200) 
	  qr_Jet_EFrac[jtyp][0] = 1;
	else if (QReport_JetEFrac[jtyp][0]->getStatus()==300) 
	  qr_Jet_EFrac[jtyp][0] = 0;
	else
	  qr_Jet_EFrac[jtyp][0] = -1;
      }
      else{ qr_Jet_EFrac[jtyp][0] = -2;
      }
      
      if (QReport_JetEFrac[jtyp][1]){
	if (QReport_JetEFrac[jtyp][1]->getStatus()==100 ||
	    QReport_JetEFrac[jtyp][1]->getStatus()==200) 
	  qr_Jet_EFrac[jtyp][1] = 1;
	else if (QReport_JetEFrac[jtyp][1]->getStatus()==300) 
	  qr_Jet_EFrac[jtyp][1] = 0;
	else
	  qr_Jet_EFrac[jtyp][1] = -1;
      }
      else{ qr_Jet_EFrac[jtyp][1] = -2;
      }
    }
    else {
      for (int ii = 0; ii < 2; ++ii) {
	//Jet NTracks test
	if (QReport_JetNTracks[ii]){
	  if (QReport_JetNTracks[ii]->getStatus()==100 ||
	      QReport_JetNTracks[ii]->getStatus()==200) 
	    qr_Jet_NTracks[ii] = 1;
	  else if (QReport_JetNTracks[ii]->getStatus()==300) 
	    qr_Jet_NTracks[ii] = 0;
	  else
	    qr_Jet_NTracks[ii] = -1;
	}
	else{ qr_Jet_NTracks[ii] = -2;
	}
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

    //Only apply certain tests, as defined in the config
    for (int ttyp = 0; ttyp < 2;  ++ttyp) {
      if (!jetTests[jtyp][ttyp]) {
	qr_Jet_Pt[jtyp][ttyp]           = 1;
	if (ttyp ==1) {
	  qr_Jet_Eta[jtyp]          = 1;
	  qr_Jet_Phi[jtyp]          = 1;
	}
	if (jtyp < 4) {
	  qr_Jet_EFrac[jtyp][ttyp]        = 1;
	  qr_Jet_Constituents[jtyp][ttyp] = 1;
	}
	else{
	  qr_Jet_NTracks[ttyp] = 1;
	}
      }
    }
    
    
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
      else if ( (qr_Jet_EFrac[jtyp][0]   == -2) &&
	   (qr_Jet_EFrac[jtyp][1]        == -2) &&
	   (qr_Jet_Constituents[jtyp][1] == -2) && 
	   (qr_Jet_Constituents[jtyp][0] == -2) &&
	   (qr_Jet_Eta[jtyp]             == -2) &&
	   (qr_Jet_Phi[jtyp]             == -2) &&
	   (qr_Jet_Pt[jtyp][0]           == -2) &&
	   (qr_Jet_Pt[jtyp][1]           == -2)
	   )
	dc_Jet[jtyp] = -2;
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
      else if ( (qr_Jet_NTracks[0] == -2) &&
	   (qr_Jet_NTracks[1]      == -2) &&
	   (qr_Jet_Eta[jtyp]       == -2) &&
	   (qr_Jet_Phi[jtyp]       == -2) &&
	   (qr_Jet_Pt[jtyp][0]     == -2) &&
	   (qr_Jet_Pt[jtyp][1]     == -2)
	   )
	dc_Jet[jtyp] = -2;
      else
	dc_Jet[jtyp] = 1;
    }
    
    if (verbose_) std::cout<<"Certifying Jet algo: "<<jtyp<<" with value: "<<dc_Jet[jtyp]<<std::endl;
    CertificationSummaryMap->Fill(2, 4-jtyp, dc_Jet[jtyp]);
    reportSummaryMap->Fill(2, 4-jtyp, dc_Jet[jtyp]);
  }

  //-----------------------------
  // MET DQM Data Certification
  //-----------------------------
  //
  // Prepare test histograms
  //
  MonitorElement *meMExy[4][2];
  MonitorElement *meMEt[4];
  MonitorElement *meSumEt[4];
  MonitorElement *meMETPhi[4];
  //MonitorElement *meMETEMFrac[5];
  //MonitorElement *meMETEmEt[3][2];
  //MonitorElement *meMETHadEt[3][2];
 
  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/MET/";
  else              newHistoName = RunDir+"/JetMET/Runsummary/MET/";

  if (isData){ //directory should be present in MC as well
    metFolder = "Cleaned";
  }else{
    metFolder   = "Uncleaned";
  }
  //MEx/MEy monitor elements
  meMExy[0][0] = dbe_->get(newHistoName+"met/"+metFolder+"/MEx");
  meMExy[0][1] = dbe_->get(newHistoName+"met/"+metFolder+"/MEy");
  meMExy[1][0] = dbe_->get(newHistoName+"pfMet/"+metFolder+"/MEx");
  meMExy[1][1] = dbe_->get(newHistoName+"pfMet/"+metFolder+"/MEy");
  meMExy[2][0] = dbe_->get(newHistoName+"tcMet/"+metFolder+"/MEx");
  meMExy[2][1] = dbe_->get(newHistoName+"tcMet/"+metFolder+"/MEy");
 
  //MET Phi monitor elements
  meMETPhi[0]  = dbe_->get(newHistoName+"met/"+metFolder+"/METPhi");
  meMETPhi[1]  = dbe_->get(newHistoName+"pfMet/"+metFolder+"/METPhi");
  meMETPhi[2]  = dbe_->get(newHistoName+"tcMet/"+metFolder+"/METPhi");
  //MET monitor elements
  meMEt[0]  = dbe_->get(newHistoName+"met/"+metFolder+"/MET");
  meMEt[1]  = dbe_->get(newHistoName+"pfMet/"+metFolder+"/MET");
  meMEt[2]  = dbe_->get(newHistoName+"tcMet/"+metFolder+"/MET");
  //SumET monitor elements
  meSumEt[0]  = dbe_->get(newHistoName+"met/"+metFolder+"/SumET");
  meSumEt[1]  = dbe_->get(newHistoName+"pfMet/"+metFolder+"/SumET");
  meSumEt[2]  = dbe_->get(newHistoName+"tcMet/"+metFolder+"/SumET");
				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------

  // 3 types of MET {CaloMET, PfMET, TcMET}  // It is 5 if CaloMETNoHF is included
  // 2 types of tests Mean test/Kolmogorov test
  const QReport * QReport_MExy[3][2][2]={{{0}}};
  const QReport * QReport_MEt[3][2]={{0}};
  const QReport * QReport_SumEt[3][2]={{0}};
  //2 types of tests phiQTest and Kolmogorov test
  const QReport * QReport_METPhi[3][2]={{0}};


  float qr_MET_MExy[3][2][2] = {{{-999.}}};
  float qr_MET_MEt[3][2]     = {{-999.}};
  float qr_MET_SumEt[3][2]   = {{-999.}};
  float qr_MET_METPhi[3][2]  = {{-999.}};
  float dc_MET[3]            = {-999.};


  // J.Piedra, 27/02/212
  // removed MuCorrMET --> loop up to 3 instead of 4, remove already from definition
  for (int mtyp = 0; mtyp < 3; ++mtyp){
    //std::cout<<"METType "<<mtyp<<std::endl;
    //Mean test results
    //std::cout<<"meMEx = :"<<meMExy[mtyp][0]<<std::endl;
    //std::cout<<"meMEy = :"<<meMExy[mtyp][1]<<std::endl;
    //std::cout<<"meMET = :"<<meMEt[mtyp]<<std::endl;
    //std::cout<<"meMETPhi = :"<<meMExy[mtyp]<<std::endl;
    //std::cout<<"meSumEt = :"<<meMExy[mtyp]<<std::endl;
    if (meMExy[mtyp][0] && meMExy[mtyp][0]->getRootObject()) {
      QReport_MExy[mtyp][0][0] = meMExy[mtyp][0]->getQReport("meanMExyTest");
      QReport_MExy[mtyp][1][0] = meMExy[mtyp][0]->getQReport("KolmogorovTest");
    }
    if (meMExy[mtyp][1]&& meMExy[mtyp][1]->getRootObject()) {
      QReport_MExy[mtyp][0][1] = meMExy[mtyp][1]->getQReport("meanMExyTest");
      QReport_MExy[mtyp][1][1] = meMExy[mtyp][1]->getQReport("KolmogorovTest");
    }
    if (meMEt[mtyp] && meMEt[mtyp]->getRootObject()) {
      QReport_MEt[mtyp][0]     = meMEt[mtyp]->getQReport("meanMETTest");
      QReport_MEt[mtyp][1]     = meMEt[mtyp]->getQReport("KolmogorovTest");
    }

    if (meSumEt[mtyp] && meSumEt[mtyp]->getRootObject()) {
      QReport_SumEt[mtyp][0]   = meSumEt[mtyp]->getQReport("meanSumETTest");
      QReport_SumEt[mtyp][1]   = meSumEt[mtyp]->getQReport("KolmogorovTest");
    }

    if (meMETPhi[mtyp] && meMETPhi[mtyp]->getRootObject()) {
      QReport_METPhi[mtyp][0]  = meMETPhi[mtyp]->getQReport("phiQTest");
      QReport_METPhi[mtyp][1]  = meMETPhi[mtyp]->getQReport("KolmogorovTest");
    }    
    for (int testtyp = 0; testtyp < 2; ++testtyp) {
      //MEx test
      if (QReport_MExy[mtyp][testtyp][0]){
	if (QReport_MExy[mtyp][testtyp][0]->getStatus()==100 ||
	    QReport_MExy[mtyp][testtyp][0]->getStatus()==200) 
	  qr_MET_MExy[mtyp][testtyp][0] = 1;
	else if (QReport_MExy[mtyp][testtyp][0]->getStatus()==300) 
	  qr_MET_MExy[mtyp][testtyp][0] = 0;
	else
	  qr_MET_MExy[mtyp][testtyp][0] = -1;
      }
      else qr_MET_MExy[mtyp][testtyp][0] = -2;
      //MEy test
      if (QReport_MExy[mtyp][testtyp][1]){
	if (QReport_MExy[mtyp][testtyp][1]->getStatus()==100 ||
	    QReport_MExy[mtyp][testtyp][1]->getStatus()==200) 
	  qr_MET_MExy[mtyp][testtyp][1] = 1;
	else if (QReport_MExy[mtyp][testtyp][1]->getStatus()==300) 
	  qr_MET_MExy[mtyp][testtyp][1] = 0;
	else
	  qr_MET_MExy[mtyp][testtyp][1] = -1;
      }
      else qr_MET_MExy[mtyp][testtyp][1] = -2;
      
      //MEt test
      if (QReport_MEt[mtyp][testtyp]){
	if (QReport_MEt[mtyp][testtyp]->getStatus()==100 ||
	    QReport_MEt[mtyp][testtyp]->getStatus()==200) 
	  qr_MET_MEt[mtyp][testtyp] = 1;
	else if (QReport_MEt[mtyp][testtyp]->getStatus()==300) 
	  qr_MET_MEt[mtyp][testtyp] = 0;
	else
	  qr_MET_MEt[mtyp][testtyp] = -1;
      }
      else{
	qr_MET_MEt[mtyp][testtyp] = -2;
      }
      //SumEt test
      if (QReport_SumEt[mtyp][testtyp]){
	if (QReport_SumEt[mtyp][testtyp]->getStatus()==100 ||
	    QReport_SumEt[mtyp][testtyp]->getStatus()==200) 
	  qr_MET_SumEt[mtyp][testtyp] = 1;
	else if (QReport_SumEt[mtyp][testtyp]->getStatus()==300) 
	  qr_MET_SumEt[mtyp][testtyp] = 0;
	else
	  qr_MET_SumEt[mtyp][testtyp] = -1;
      }
      else{
	qr_MET_SumEt[mtyp][testtyp] = -2;
      }
      //METPhi test
      if (QReport_METPhi[mtyp][testtyp]){
	if (QReport_METPhi[mtyp][testtyp]->getStatus()==100 ||
	    QReport_METPhi[mtyp][testtyp]->getStatus()==200) 
	  qr_MET_METPhi[mtyp][testtyp] = 1;
	else if (QReport_METPhi[mtyp][testtyp]->getStatus()==300) 
	  qr_MET_METPhi[mtyp][testtyp] = 0;
	else
	  qr_MET_METPhi[mtyp][testtyp] = -1;
      }
      else{
	qr_MET_METPhi[mtyp][testtyp] = -2;
      }
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


    //Only apply certain tests, as defined in the config
    for (int ttyp = 0; ttyp < 2;  ++ttyp) {
      if (!metTests[mtyp][ttyp]) {
	qr_MET_MExy[mtyp][ttyp][0]   = 1;
	qr_MET_MExy[mtyp][ttyp][1]   = 1;
	qr_MET_MEt[mtyp][ttyp]       = 1;
	qr_MET_SumEt[mtyp][ttyp]     = 1;
	qr_MET_METPhi[mtyp][ttyp]    = 1;
      }
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
    else if ( 
	(qr_MET_MExy[mtyp][0][0] == -2) &&
	(qr_MET_MExy[mtyp][0][1] == -2) &&
	(qr_MET_MEt[mtyp][0]     == -2) &&
	(qr_MET_SumEt[mtyp][0]   == -2) &&
	(qr_MET_METPhi[mtyp][0]  == -2) &&
	(qr_MET_MExy[mtyp][1][0] == -2) &&
	(qr_MET_MExy[mtyp][1][1] == -2) &&
	(qr_MET_MEt[mtyp][1]     == -2) &&
	(qr_MET_SumEt[mtyp][1]   == -2) &&
	(qr_MET_METPhi[mtyp][1]  == -2)
	)
      dc_MET[mtyp] = -2;
    else
      dc_MET[mtyp] = 1;

    if (verbose_) std::cout<<"Certifying MET algo: "<<mtyp<<" with value: "<<dc_MET[mtyp]<<std::endl;
    CertificationSummaryMap->Fill(1, 4-mtyp, dc_MET[mtyp]);
    reportSummaryMap->Fill(1, 4-mtyp, dc_MET[mtyp]);
  }


  //-----------------------------
  // CaloTowers DQM Data Certification
  //-----------------------------

  //
  // Prepare test histograms
  //
  //MonitorElement *meCTOcc[3];
  //MonitorElement *meCTEn[3];
  //MonitorElement *meCT[3];
  //MonitorElement *meCT[3];
 
  //RunDir = "";
  //if (RunDir == "") newHistoName = "JetMET/MET/";
  //else              newHistoName = RunDir+"/JetMET/Run summary/MET/";

  //meMExy[0][0] = dbe_->get(newHistoName+"CaloMET/"+cleaningdir+"/"+metFolder+"/METTask_CaloMEx");
  //meMExy[0][1] = dbe_->get(newHistoName+"CaloMET/"+cleaningdir+"/"+metFolder+"/METTask_CaloMEy");
  //meMExy[1][0] = dbe_->get(newHistoName+"CaloMETNoHF/"+cleaningdir+"/"+metFolder+"/METTask_CaloMEx");
				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------
  // Commenting out unused but initialized variables. [Suchandra Dutta]
  //  float qr_CT_Occ[3] = {-2.};
  float dc_CT[3]     = {-2.};
  dc_CT[0]  = -2.;
  dc_CT[1]  = -2.;
  dc_CT[2]  = -2.;

  //  qr_CT_Occ[0]  = dc_CT[0];
  //  qr_CT_Occ[1]  = dc_CT[1];
  //  qr_CT_Occ[2]  = dc_CT[2];

  for (int cttyp = 0; cttyp < 3; ++cttyp) {
    
    if (verbose_) std::cout<<"Certifying CaloTowers with value: "<<dc_CT[cttyp]<<std::endl;
    CertificationSummaryMap->Fill(0, 4-cttyp, dc_CT[cttyp]);
    reportSummaryMap->Fill(0, 4-cttyp, dc_CT[cttyp]);
  }
  dbe_->setCurrentFolder("");  
}

//define this as a plug-in
//DEFINE_FWK_MODULE(DataCertificationJetMET);
