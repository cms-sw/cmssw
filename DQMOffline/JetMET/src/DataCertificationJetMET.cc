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
 
  if (verbose_) std::cout << ">>> Constructor (DataCertificationJetMET) <<<" << std::endl;

  // -----------------------------------------
  //
}


DataCertificationJetMET::~DataCertificationJetMET()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (verbose_) std::cout << ">>> Deconstructor (DataCertificationJetMET) <<<" << std::endl;
}


// ------------ method called right after a run ends ------------
void 
DataCertificationJetMET::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_)
{
  if (verbose_) std::cout << ">>> EndRun (DataCertificationJetMET) <<<" << std::endl;

  std::vector<std::string> subDirVec;
  std::string RunDir;
  std::string RunNum;

  std::string RefRunDir;

  if (verbose_) std::cout << "InMemory_           = " << InMemory_    << std::endl;

  ibook_.setCurrentFolder(folderName);  
  reportSummary = ibook_.bookFloat("reportSummary");
  CertificationSummary = ibook_.bookFloat("CertificationSummary");
  
  reportSummaryMap = ibook_.book2D("reportSummaryMap","reportSummaryMap",3,0,3,5,0,5);
  CertificationSummaryMap = ibook_.book2D("CertificationSummaryMap","CertificationSummaryMap",3,0,3,5,0,5);


  reportSummary = iget_.get(folderName+"/"+"reportSummary");
  CertificationSummary = iget_.get(folderName+"/"+"CertificationSummary");
  reportSummaryMap = iget_.get(folderName+"/"+"reportSummaryMap");
  CertificationSummaryMap = iget_.get(folderName+"/"+"CertificationSummaryMap");


  
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
  ibook_.setCurrentFolder("JetMET/EventInfo/CertificationSummaryContents/");    


  std::string refHistoName;
  std::string newHistoName;
  
  //-----------------------------
  // Jet DQM Data Certification
  //-----------------------------
  //we have 4 types anymore: PF (barrel,endcap,forward) and calojets
  MonitorElement *meJetPt[4];
  MonitorElement *meJetEta[4];
  MonitorElement *meJetPhi[4];
  MonitorElement *meJetEMFrac[4];
  MonitorElement *meJetConstituents[4];
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
  meJetPhi[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_Barrel");
  meJetPhi[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_EndCap");
  meJetPhi[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_Forward");
  meJetPhi[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Phi");
  //meJetPhi[4] = iget_.get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Phi");

  //Jet Eta histos
  meJetEta[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Eta");
  meJetEta[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Eta");
  meJetEta[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EtaFirst");
  meJetEta[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Eta");
  //meJetEta[4] = iget_.get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Eta");

  //Jet Pt histos
  meJetPt[0]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_Barrel");
  meJetPt[1]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_EndCap");
  meJetPt[2]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_Forward");
  meJetPt[3]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Pt_2");

  ////Jet Constituents histos
  meJetConstituents[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Barrel");
  meJetConstituents[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_EndCap");
  meJetConstituents[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Forward");
  meJetConstituents[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Constituents");
  //
  ////Jet EMFrac histos
  meJetEMFrac[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Barrel");
  meJetEMFrac[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_EndCap");
  meJetEMFrac[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Forward");
  meJetEMFrac[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/EFrac");

				   
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


  // Four types of jets {AK5 Barrel, AK5 EndCap, AK5 Forward, PF}, removed JPT which is 5th type of jets
  //----------------------------------------------------------------------------
  // Kolmogorov (KS) tests
  const QReport* QReport_JetEta[4] = {0};
  const QReport* QReport_JetPhi[4] = {0};
  // Mean and KS tests for Calo and PF jets
  const QReport* QReport_JetConstituents[4][2] = {{0}};
  const QReport* QReport_JetEFrac[4][2]        = {{0}};
  const QReport* QReport_JetPt[4][2]           = {{0}};

  // Mean and KS tests for JPT jets
  //const QReport* QReport_JetNTracks[2] = {0, 0};
  float qr_Jet_Eta[4]     = {-1};
  float qr_Jet_Phi[4]     = {-1};
  float dc_Jet[4]         = {-1};

  float qr_Jet_Constituents[4][2] = {{-1}};
  float qr_Jet_EFrac[4][2]        = {{-1}};
  float qr_Jet_Pt[4][2]           = {{-1}};


  // Loop
  //----------------------------------------------------------------------------
  for (int jtyp=0; jtyp<4; ++jtyp) {
    // Mean test results

    if (meJetConstituents[jtyp] && meJetConstituents[jtyp]->getRootObject() ) {
      QReport_JetConstituents[jtyp][0] = meJetConstituents[jtyp]->getQReport("meanJetConstituentsTest");
      QReport_JetConstituents[jtyp][1] = meJetConstituents[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetEMFrac[jtyp]&& meJetEMFrac[jtyp]->getRootObject() ) {
      QReport_JetEFrac[jtyp][0]        = meJetEMFrac[jtyp]->getQReport("meanEMFractionTest");
      QReport_JetEFrac[jtyp][1]        = meJetEMFrac[jtyp]->getQReport("KolmogorovTest");
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
    
    if (verbose_) {
      printf("====================Jet Type %d QTest Report Summary========================\n",jtyp);
      printf("Eta:    Phi:   Pt 1:    2:    Const/Ntracks 1:    2:    EFrac/tracknhits 1:    2:\n");

      printf("%2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f\n", \
	     qr_Jet_Eta[jtyp],						\
	     qr_Jet_Phi[jtyp],						\
	     qr_Jet_Pt[jtyp][0],					\
	     qr_Jet_Pt[jtyp][1],					\
	     qr_Jet_Constituents[jtyp][0],				\
	     qr_Jet_Constituents[jtyp][1],				\
	     qr_Jet_EFrac[jtyp][0],					\
	     qr_Jet_EFrac[jtyp][1]);
      
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
	qr_Jet_EFrac[jtyp][ttyp]        = 1;
	qr_Jet_Constituents[jtyp][ttyp] = 1;
      }
    }
    
    
  
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
  MonitorElement *meMExy[2][2];
  MonitorElement *meMEt[2];
  MonitorElement *meSumEt[2];
  MonitorElement *meMETPhi[2];
 
  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/MET/";
  else              newHistoName = RunDir+"/JetMET/Runsummary/MET/";

  if (isData){ //directory should be present in MC as well
    metFolder = "Cleaned";
  }else{
    metFolder   = "Uncleaned";
  }
  //MEx/MEy monitor elements
  meMExy[0][0] = iget_.get(newHistoName+"met/"+metFolder+"/MEx");
  meMExy[0][1] = iget_.get(newHistoName+"met/"+metFolder+"/MEy");
  meMExy[1][0] = iget_.get(newHistoName+"pfMet/"+metFolder+"/MEx");
  meMExy[1][1] = iget_.get(newHistoName+"pfMet/"+metFolder+"/MEy");
 
  //MET Phi monitor elements
  meMETPhi[0]  = iget_.get(newHistoName+"met/"+metFolder+"/METPhi");
  meMETPhi[1]  = iget_.get(newHistoName+"pfMet/"+metFolder+"/METPhi");
  //MET monitor elements
  meMEt[0]  = iget_.get(newHistoName+"met/"+metFolder+"/MET");
  meMEt[1]  = iget_.get(newHistoName+"pfMet/"+metFolder+"/MET");
  //SumET monitor elements
  meSumEt[0]  = iget_.get(newHistoName+"met/"+metFolder+"/SumET");
  meSumEt[1]  = iget_.get(newHistoName+"pfMet/"+metFolder+"/SumET");
				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------

  // 2 types of MET {CaloMET, PfMET}  // It is 5 if CaloMETNoHF is included, 4 for MuonCorMET
  // removed 3rd type of TcMET
  // 2 types of tests Mean test/Kolmogorov test
  const QReport * QReport_MExy[2][2][2]={{{0}}};
  const QReport * QReport_MEt[2][2]={{0}};
  const QReport * QReport_SumEt[2][2]={{0}};
  //2 types of tests phiQTest and Kolmogorov test
  const QReport * QReport_METPhi[2][2]={{0}};


  float qr_MET_MExy[2][2][2] = {{{-999.}}};
  float qr_MET_MEt[2][2]     = {{-999.}};
  float qr_MET_SumEt[2][2]   = {{-999.}};
  float qr_MET_METPhi[2][2]  = {{-999.}};
  float dc_MET[2]            = {-999.};


  // J.Piedra, 27/02/212
  // removed MuCorrMET & TcMET --> loop up to 2 instead of 4, remove already from definition
  for (int mtyp = 0; mtyp < 2; ++mtyp){
    //Mean test results
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

				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------
  // Commenting out unused but initialized variables. [Suchandra Dutta]
  float dc_CT[3]     = {-2.};
  dc_CT[0]  = -2.;
  dc_CT[1]  = -2.;
  dc_CT[2]  = -2.;

  for (int cttyp = 0; cttyp < 3; ++cttyp) {
    
    if (verbose_) std::cout<<"Certifying CaloTowers with value: "<<dc_CT[cttyp]<<std::endl;
    CertificationSummaryMap->Fill(0, 4-cttyp, dc_CT[cttyp]);
    reportSummaryMap->Fill(0, 4-cttyp, dc_CT[cttyp]);
  }
  ibook_.setCurrentFolder("");  
}

//define this as a plug-in
//DEFINE_FWK_MODULE(DataCertificationJetMET);
