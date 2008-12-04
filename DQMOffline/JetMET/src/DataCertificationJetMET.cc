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
// $Id: DataCertificationJetMET.cc,v 1.11 2008/10/15 18:42:01 chlebana Exp $
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

#define NJetAlgo 4
#define NL3Flags 3

#define DEBUG    1

// #include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"


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

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   DQMStore * dbe;
   DQMStore * rdbe;
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


// ------------------------------------------------------------
int data_certificate(double chi2, double mean, double chi2_tolerance, double mean_tolerance){
  int value=0;
  if (chi2<chi2_tolerance && fabs(mean)<mean_tolerance) value=1;
  return value;
}

// ------------------------------------------------------------
void
fitd(TH1F* hist, TF1* fn, TF1* f1, TF1* f2, int verbose){
  //
  Option_t *fit_option;
  fit_option = "QR0";
  if      (verbose==1) fit_option = "R0";
  else if (verbose==2) fit_option = "VR0";
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
fitdd(TH1D* hist, TF1* fn, TF1* f1, TF1* f2, int verbose){
  //
  Option_t *fit_option;
  fit_option = "QR0";
  if      (verbose==1) fit_option = "R0";
  else if (verbose==2) fit_option = "VR0";
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

  std::string filename    = conf_.getUntrackedParameter<std::string>("fileName");
  std::string reffilename = conf_.getUntrackedParameter<std::string>("refFileName");
  std::cout << "Reference FileName = " << reffilename << std::endl;
  std::cout << "FileName           = " << filename    << std::endl;

  // -- Current Run
  dbe = edm::Service<DQMStore>().operator->();
  dbe->open(filename);

  dbe->setCurrentFolder("/");
  std::string currDir = dbe->pwd();
  std::cout << "--- Current Directory " << currDir << std::endl;

  std::vector<std::string> subDirVec = dbe->getSubdirs();

  std::string RunDir;
  std::string RunNum;
  int         RunNumber;

  // TODO: Make sure this is correct....
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    RunDir = *ic;
    RunNum = *ic;
  }
  RunNum.erase(0,4);
  RunNumber = atoi(RunNum.c_str());
  std::cout << "--- >>" << RunNumber << "<<" << std::endl;

  // --- Reference set of histograms
  rdbe = edm::Service<DQMStore>().operator->();
  rdbe->open(reffilename);

  std::vector<MonitorElement*> mes = dbe->getAllContents("");
  std::cout << "found " << mes.size() << " monitoring elements!" << std::endl;

  // TH1F *bla = fs_->make<TH1F>("bla","bla",256,0,256);
  // int totF;

  //
  // Data certification starts
  //----------------------------------------------------------------

  const int nLSBins=500;

  //Double_t par[6];
  //double chindf[6];
  //double para[6][6];

  TF1 *g1    = new TF1("g1","gaus",-50,50);
  TF1 *g2    = new TF1("g2","gaus",-500,500);
  TF1 *dgaus = new TF1("dgaus","gaus(0)+gaus(3)",-500,500);
  //TF1 *sgaus = new TF1("sgaus","gaus",-500,500);

  TH1F *hMExy[6];
  TF1  *fitfun[6];
  TF1  *fitfun1[6];
  TF1  *fitfun2[6];
  TH2F *hCaloMEx_LS;
  TH2F *hCaloMEy_LS;
  TH2F *hCaloMExNoHF_LS;
  TH2F *hCaloMEyNoHF_LS;


  // ****************************
  // ****************************

  // --- Save Data Certification results to the root file
  //     We save both flags and values
  dbe->setCurrentFolder(RunDir+"/JetMET/Data Certification/");    
  MonitorElement* mJetDCFL1 = dbe->book1D("JetDCFLayer1", "Jet DC F L1", 1, 0, 1);
  MonitorElement* mJetDCFL2 = dbe->book1D("JetDCFLayer2", "Jet DC F L2", NJetAlgo, 0, NJetAlgo);
  MonitorElement* mJetDCFL3 = dbe->book1D("JetDCFLayer3", "Jet DC F L3", NJetAlgo*NL3Flags, 0, NJetAlgo*NL3Flags);

  MonitorElement* mJetDCVL1 = dbe->book1D("JetDCVLayer1", "Jet DC V L1", NJetAlgo, 0, NJetAlgo);
  MonitorElement* mJetDCVL2 = dbe->book1D("JetDCVLayer2", "Jet DC V L2", NJetAlgo, 0, NJetAlgo);
  MonitorElement* mJetDCVL3 = dbe->book1D("JetDCVLayer3", "Jet DC V L3", NJetAlgo*NL3Flags, 0, NJetAlgo*NL3Flags);

  MonitorElement* mMETDCFL1 = dbe->book2D("METDCFLayer1", "MET DC F L1", 3,0,3,500,0.,500.);
  MonitorElement* mMETDCFL2 = dbe->book2D("METDCFLayer2", "MET DC F L2", 3,0,3,500,0.,500.);
  MonitorElement* mMETDCFL3 = dbe->book2D("METDCFLayer3", "MET DC F L3", 3,0,3,500,0.,500.);

  MonitorElement* mMETDCVL1 = dbe->book2D("METDCVLayer1", "MET DC V L1", 3,0,3,500,0.,500.);
  MonitorElement* mMETDCVL2 = dbe->book2D("METDCVLayer2", "MET DC V L2", 3,0,3,500,0.,500.);
  MonitorElement* mMETDCVL3 = dbe->book2D("METDCVLayer3", "MET DC V L3", 3,0,3,500,0.,500.);


  Double_t test_Pt, test_Eta, test_Phi, test_Constituents, test_HFrac;
  test_Pt = test_Eta = test_Phi = test_Constituents = test_HFrac = 0;
  
  Double_t test_Pt_Barrel,  test_Phi_Barrel;
  Double_t test_Pt_EndCap,  test_Phi_EndCap;
  Double_t test_Pt_Forward, test_Phi_Forward;
  test_Pt_Barrel  = test_Phi_Barrel  = 0;
  test_Pt_EndCap  = test_Phi_EndCap  = 0;
  test_Pt_Forward = test_Phi_Forward = 0;

  Int_t Jet_DCF_L1[NJetAlgo];
  Int_t Jet_DCF_L2[NJetAlgo];
  Int_t Jet_DCF_L3[NJetAlgo][NL3Flags];

  //  Int_t Jet_DCV_L1[NJetAlgo];
  //  Int_t Jet_DCV_L2[NJetAlgo];
  //  Int_t Jet_DCV_L3[NJetAlgo][NL3Flags];

  //  Int_t Jet_DC[NJetAlgo];
  std::string Jet_Tag_L1[2];
  Jet_Tag_L1[0]    = "JetMET_Jet";
  Jet_Tag_L1[1]    = "JetMET_MET";

  std::string Jet_Tag_L2[NJetAlgo];
  Jet_Tag_L2[0] = "JetMET_Jet_IterativeCone";
  Jet_Tag_L2[1] = "JetMET_Jet_SISCone";
  Jet_Tag_L2[2] = "JetMET_Jet_PFlow";
  Jet_Tag_L2[3] = "JetMET_Jet_JPT";

  std::string Jet_Tag_L3[NJetAlgo][NL3Flags];
  Jet_Tag_L3[0][0] = "JetMET_Jet_IterativeCone_Barrel";
  Jet_Tag_L3[0][1] = "JetMET_Jet_IterativeCone_EndCap";
  Jet_Tag_L3[0][2] = "JetMET_Jet_IterativeCone_Forward";
  Jet_Tag_L3[1][0] = "JetMET_Jet_SISCone_Barrel";
  Jet_Tag_L3[1][1] = "JetMET_Jet_SISCone_EndCap";
  Jet_Tag_L3[1][2] = "JetMET_Jet_SISCone_Forward";
  Jet_Tag_L3[2][0] = "JetMET_Jet_PFlow_Barrel";
  Jet_Tag_L3[2][1] = "JetMET_Jet_PFlow_EndCap";
  Jet_Tag_L3[2][2] = "JetMET_Jet_PFlow_Forward";
  Jet_Tag_L3[3][0] = "JetMET_Jet_JPT_Barrel";
  Jet_Tag_L3[3][1] = "JetMET_Jet_JPT_EndCap";
  Jet_Tag_L3[3][2] = "JetMET_Jet_JPT_Forward";
  
  //  rdbe->setCurrentFolder(RunDir+"/JetMET/Run summary/SISConeJets");
  //  std::string refHistoName = RunDir+"/JetMET/Run summary/PFJetAnalyzer/Pt";

  std::string refHistoName;
  std::string newHistoName;

  // --- Loop over jet algorithms for Layer 2
  for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    

    if (iAlgo == 0) {
      refHistoName = RunDir+"/JetMET/Run summary/IterativeConeJets/";
      //      newHistoName = RunDir+"/JetMET/Run summary/IterativeConeJets/";
      newHistoName = RunDir+"/JetMET/Run summary/SISConeJets/";
    }
    if (iAlgo == 1) {
      refHistoName = RunDir+"/JetMET/Run summary/SISConeJets/";
      newHistoName = RunDir+"/JetMET/Run summary/SISConeJets/";
    }
    if (iAlgo == 2) {
      refHistoName = RunDir+"/JetMET/Run summary/PFJets/";
      newHistoName = RunDir+"/JetMET/Run summary/PFJets/";
    }
    if (iAlgo == 3) {
      refHistoName = RunDir+"/JetMET/Run summary/JPT/";
      newHistoName = RunDir+"/JetMET/Run summary/JPT/";
    }



    // ----------------
    // --- Layer 2
    MonitorElement * meRef = rdbe->get(refHistoName+"Pt");
    MonitorElement * meNew = dbe->get(newHistoName+"Pt");
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

    meRef = rdbe->get(refHistoName+"Eta");
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

    meRef = rdbe->get(refHistoName+"Phi");
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
     
    meRef = rdbe->get(refHistoName+"Constituents");
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
     
    meRef = rdbe->get(refHistoName+"EnergyFractionHadronic");
    meNew = dbe->get(newHistoName+"EnergyFractionHadronic");
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

    if ( (test_Pt     > 0.95) && (test_Eta          > 0.95) && 
	 (test_Phi    > 0.95) && (test_Constituents > 0.95) && 
	 (test_HFrac  > 0.95) )  {      

      Jet_DCF_L2[iAlgo] = 1;
      // --- Fill DC results histogram
      mJetDCFL2->Fill(iAlgo);
    } else {
      Jet_DCF_L2[iAlgo] = 0;
    }

    // ----------------
    // --- Layer 3
    // --- Barrel
    meRef = rdbe->get(refHistoName+"Pt_Barrel");
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

    meRef = rdbe->get(refHistoName+"Phi_Barrel");
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
    meRef = rdbe->get(refHistoName+"Pt_EndCap");
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

    meRef = rdbe->get(refHistoName+"Phi_EndCap");
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
    meRef = rdbe->get(refHistoName+"Pt_Forward");
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

    meRef = rdbe->get(refHistoName+"Phi_Forward");
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


    if ( (test_Pt_Barrel > 0.95) && (test_Phi_Barrel > 0.95) ) {
      Jet_DCF_L3[iAlgo][0] = 1;
      // --- Fill DC results histogram
      mJetDCFL3->Fill(iAlgo+0*NL3Flags);
    } else {
      Jet_DCF_L3[iAlgo][0] = 0;
    }
    if ( (test_Pt_EndCap > 0.95) && (test_Phi_EndCap > 0.95) ) {
      Jet_DCF_L3[iAlgo][1] = 1;
      // --- Fill DC results histogram
      mJetDCFL3->Fill(iAlgo+1*NL3Flags);
    } else {
      Jet_DCF_L3[iAlgo][1] = 0;
    }
    if ( (test_Pt_Forward > 0.95) && (test_Phi_Forward > 0.95) ) {
      Jet_DCF_L3[iAlgo][2] = 1;
      // --- Fill DC results histogram
      mJetDCFL3->Fill(iAlgo+2*NL3Flags);
    } else {
      Jet_DCF_L3[iAlgo][2] = 0;
    }

  }

  // --- End of loop over jet algorithms
  int allOK = 1;
  for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {   
    if (Jet_DCF_L1[iAlgo] == 0) allOK = 0;
  }
  if (allOK == 1) mJetDCFL1->Fill(1);


  // JET Data Certification Results
  if (DEBUG) {
    std::cout << std::endl;
    printf("%6d %15d %-35s %10d\n",RunNumber,0,"JetMET", allOK);
    for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    
      printf("%6d %15d %-35s %10d\n",RunNumber,0,Jet_Tag_L2[iAlgo].c_str(), Jet_DCF_L2[iAlgo]);
    }
    for (int iAlgo=0; iAlgo<NJetAlgo; iAlgo++) {    
      for (int iL3Flag=0; iL3Flag<NL3Flags; iL3Flag++) {    
	printf("%6d %15d %-35s %10d\n",RunNumber,0,Jet_Tag_L3[iAlgo][iL3Flag].c_str(), Jet_DCF_L3[iAlgo][iL3Flag]);
      }
    }
    std::cout << std::endl;    
  }

  // ****************************
  // Loop over Monitoring Elements and fill working histograms
  for(std::vector<MonitorElement*>::const_iterator ime = mes.begin(); ime!=mes.end(); ++ime) {
    std::string name = (*ime)->getName();

    if (name == "METTask_CaloMEx")     hMExy[0] = (*ime)->getTH1F();
    if (name == "METTask_CaloMEy")     hMExy[1] = (*ime)->getTH1F();
    if (name == "METTask_CaloMExNoHF") hMExy[2] = (*ime)->getTH1F();
    if (name == "METTask_CaloMEyNoHF") hMExy[3] = (*ime)->getTH1F();

    if (name == "METTask_CaloMEx_LS")     hCaloMEx_LS     = (*ime)->getTH2F();
    if (name == "METTask_CaloMEy_LS")     hCaloMEy_LS     = (*ime)->getTH2F();
    if (name == "METTask_CaloMExNoHF_LS") hCaloMExNoHF_LS = (*ime)->getTH2F();
    if (name == "METTask_CaloMEyNoHF_LS") hCaloMEyNoHF_LS = (*ime)->getTH2F();

  }

  for (int i=0;i<4;i++) {
    if (hMExy[i]->GetSum()>0.){
      fitd(hMExy[i],dgaus,g1,g2,verbose);
      fitfun[i]  = hMExy[i]->GetFunction("dgaus");
      fitfun1[i] = (TF1*)g1->Clone();
      fitfun2[i] = (TF1*)g2->Clone();
    }
  }

  // Slice *_LS histograms
  TH1D *CaloMEx_LS[500];
  TH1D *CaloMEy_LS[500];
  TH1D *CaloMExNoHF_LS[500];
  TH1D *CaloMEyNoHF_LS[500];
  TF1 *fitfun_CaloMEx_LS[500];
  TF1 *fitfun_CaloMEy_LS[500];
  TF1 *fitfun_CaloMExNoHF_LS[500];
  TF1 *fitfun_CaloMEyNoHF_LS[500];
  TF1 *fitfun1_CaloMEx_LS[500];
  TF1 *fitfun1_CaloMEy_LS[500];
  TF1 *fitfun1_CaloMExNoHF_LS[500];
  TF1 *fitfun1_CaloMEyNoHF_LS[500];
  TF1 *fitfun2_CaloMEx_LS[500];
  TF1 *fitfun2_CaloMEy_LS[500];
  TF1 *fitfun2_CaloMExNoHF_LS[500];
  TF1 *fitfun2_CaloMEyNoHF_LS[500];
  int JetMET_MET[500];
  int JetMET_MET_All[500];
  int JetMET_MEx_All[500];
  int JetMET_MEy_All[500];
  int JetMET_MET_NoHF[500];
  int JetMET_MEx_NoHF[500];
  int JetMET_MEy_NoHF[500];
  for (int i=0;i<500;i++){
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
  for (int LS=1; LS<500; LS++){

    // Projection returns a 
    sprintf(ctitle,"CaloMEx_%04d",LS);     CaloMEx_LS[LS]=hCaloMEx_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMEy_%04d",LS);     CaloMEy_LS[LS]=hCaloMEy_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMExNoHF_%04d",LS); CaloMExNoHF_LS[LS]=hCaloMExNoHF_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMEyNoHF_%04d",LS); CaloMEyNoHF_LS[LS]=hCaloMEyNoHF_LS->ProjectionX(ctitle,LS+1,LS+1);

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
  }


  //----------------------------------------------------------------
  //--- Print out data certification summary
  //----------------------------------------------------------------

  if (verbose) {
  std::cout << std::endl;
  printf("| Variable                       |   Reduced chi^2              | Mean               | Width      |\n");
  }
  //
  // Entire run
  for (int i=0;i<4;i++){
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
      JetMET_MEx_All[0]=data_certificate(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
                                          fitfun[i]->GetParameter(nmean),
                                          5.,10.);
    if (i==1)
      JetMET_MEy_All[0]=data_certificate(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
                                          fitfun[i]->GetParameter(nmean),
                                          5.,10.);
    if (i==2)
      JetMET_MEx_NoHF[0]=data_certificate(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
                                          fitfun[i]->GetParameter(nmean),
                                          5.,10.);
    if (i==3)
      JetMET_MEy_NoHF[0]=data_certificate(fitfun[i]->GetChisquare()/double(fitfun[i]->GetNDF()),
                                          fitfun[i]->GetParameter(nmean),
                                          5.,10.);
  }
  //
  // Each lumi section
  // (LS=0 assigned to the entire run)
  for (int LS=1; LS<500; LS++){
    if (CaloMEx_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMEx_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      if (verbose)
      printf("\n| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEx_LS[LS]->GetName(),
             fitfun_CaloMEx_LS[LS]->GetChisquare(),double(fitfun_CaloMEx_LS[LS]->GetNDF()),
             fitfun_CaloMEx_LS[LS]->GetChisquare()/double(fitfun_CaloMEx_LS[LS]->GetNDF()),
             fitfun_CaloMEx_LS[LS]->GetParameter(nmean),  fitfun_CaloMEx_LS[LS]->GetParError(nmean),
             fitfun_CaloMEx_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEx_LS[LS]->GetParError(nmean+1));
      JetMET_MEx_All[LS]=data_certificate(fitfun_CaloMEx_LS[LS]->GetChisquare()/double(fitfun_CaloMEx_LS[LS]->GetNDF()),
                                          fitfun_CaloMEx_LS[LS]->GetParameter(nmean),
                                          5.,10.);
    }
    if (CaloMEy_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMEy_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      if (verbose)
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEy_LS[LS]->GetName(),
             fitfun_CaloMEy_LS[LS]->GetChisquare(),double(fitfun_CaloMEy_LS[LS]->GetNDF()),
             fitfun_CaloMEy_LS[LS]->GetChisquare()/double(fitfun_CaloMEy_LS[LS]->GetNDF()),
             fitfun_CaloMEy_LS[LS]->GetParameter(nmean),  fitfun_CaloMEy_LS[LS]->GetParError(nmean),
             fitfun_CaloMEy_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEy_LS[LS]->GetParError(nmean+1));
      JetMET_MEy_All[LS]=data_certificate(fitfun_CaloMEy_LS[LS]->GetChisquare()/double(fitfun_CaloMEy_LS[LS]->GetNDF()),
                                          fitfun_CaloMEy_LS[LS]->GetParameter(nmean),
                                          5.,10.);
    }
    if (CaloMExNoHF_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMExNoHF_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      if (verbose)
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMExNoHF_LS[LS]->GetName(),
             fitfun_CaloMExNoHF_LS[LS]->GetChisquare(),double(fitfun_CaloMExNoHF_LS[LS]->GetNDF()),
             fitfun_CaloMExNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMExNoHF_LS[LS]->GetNDF()),
             fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean),  fitfun_CaloMExNoHF_LS[LS]->GetParError(nmean),
             fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean+1),fitfun_CaloMExNoHF_LS[LS]->GetParError(nmean+1));
      JetMET_MEx_NoHF[LS]=data_certificate(fitfun_CaloMExNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMExNoHF_LS[LS]->GetNDF()),
                                           fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean),
                                           5.,10.);
    }
    if (CaloMEyNoHF_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMEyNoHF_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      if (verbose)
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEyNoHF_LS[LS]->GetName(),
             fitfun_CaloMEyNoHF_LS[LS]->GetChisquare(),double(fitfun_CaloMEyNoHF_LS[LS]->GetNDF()),
             fitfun_CaloMEyNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMEyNoHF_LS[LS]->GetNDF()),
             fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean),  fitfun_CaloMEyNoHF_LS[LS]->GetParError(nmean),
             fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEyNoHF_LS[LS]->GetParError(nmean+1));
      JetMET_MEy_NoHF[LS]=data_certificate(fitfun_CaloMEyNoHF_LS[LS]->GetChisquare()/double(fitfun_CaloMEyNoHF_LS[LS]->GetNDF()),
                                           fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean),
                                           5.,10.);
    }
  } // loop over LS


  //
  // Data certification format
  std::cout << std::endl;
  printf("   run,       lumi-sec, tag name,                                   output\n");
  for (int LS=0; LS<nLSBins; LS++){
    JetMET_MET_All[LS] = JetMET_MEx_All[LS] * JetMET_MEy_All[LS];
    JetMET_MET_NoHF[LS]= JetMET_MEx_NoHF[LS]* JetMET_MEy_NoHF[LS];
    JetMET_MET[LS]     = JetMET_MET_All[LS] * JetMET_MET_NoHF[LS];

    // -- Fill the DC Result Histograms    
    mMETDCFL2->Fill(0,LS,JetMET_MET_All[LS]);
    mMETDCFL2->Fill(1,LS,JetMET_MET_NoHF[LS]);
    mMETDCFL2->Fill(2,LS,JetMET_MET[LS]);

    //    std::cout  << ">>> " << LS << " " << JetMET_MET_All[LS] << " " 
    //	       << JetMET_MET_NoHF[LS] << " " << JetMET_MET[LS] << std::endl;

    if (JetMET_MET[LS]>-1.) {
      if (LS==0 || LS==1){
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET",     JetMET_MET[LS]);
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
      }
      else if (JetMET_MET[LS-1]==-1.) {	
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET",     JetMET_MET[LS]);
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
      }
      else {
	if (JetMET_MET[LS]!=JetMET_MET[LS-1])
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET",     JetMET_MET[LS]);
	if (JetMET_MET_All[LS]!=JetMET_MET_All[LS-1])
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
	if (JetMET_MET_NoHF[LS]!=JetMET_MET_NoHF[LS-1])
	printf("%6d %15d %-35s %10d\n",RunNumber,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
      }
    }
  }

  std::cout << std::endl;

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

  std::cout << ">>> endJob " << outputFile << std:: endl;

  if(outputFile){
    //    dbe->showDirStructure();
    dbe->save(outputFileName);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DataCertificationJetMET);
