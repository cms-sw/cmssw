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
// $Id: DataCertificationJetMET.cc,v 1.2 2008/10/06 14:15:37 chlebana Exp $
//
//


// system include files
#include <memory>
// #include <stdio.h>
#include <math.h>


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
}


// ------------------------------------------------------------
void
fitd(TH1F* hist, TF1* fn, TF1* f1, TF1* f2){
  //
  Double_t par[6];
  //
  hist->Rebin(2);
  hist->GetXaxis()->SetRange(201,300);
  //
  hist->Fit(f2,"R");
  f2->GetParameters(&par[3]);
  fn->SetParameters(par);
  fn->FixParameter(0,0.);
  fn->FixParameter(1,0.);
  fn->FixParameter(2,0.);
  fn->SetParLimits(5,0.,1000.);
  fn->SetParName(3,"Constant");
  fn->SetParName(4,"Mean");
  fn->SetParName(5,"Sigma");
  hist->Fit(fn,"R");
  //
  double chi2=-1.;
  if (fn->GetNDF()>0.) chi2=fn->GetChisquare()/fn->GetNDF();
  if (chi2>3.){
    hist->Fit(f1,"R");
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
    hist->Fit(fn,"R");
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
fitdd(TH1D* hist, TF1* fn, TF1* f1, TF1* f2){
  //
  Double_t par[6];
  //
  hist->Rebin(2);
  hist->GetXaxis()->SetRange(201,300);
  //
  hist->Fit(f2,"R");
  f2->GetParameters(&par[3]);
  fn->SetParameters(par);
  fn->FixParameter(0,0.);
  fn->FixParameter(1,0.);
  fn->FixParameter(2,0.);
  fn->SetParLimits(5,0.,1000.);
  fn->SetParName(3,"Constant");
  fn->SetParName(4,"Mean");
  fn->SetParName(5,"Sigma");
  hist->Fit(fn,"R");
  //
  double chi2=-1.;
  if (fn->GetNDF()>0.) chi2=fn->GetChisquare()/fn->GetNDF();
  if (chi2>3.){
    hist->Fit(f1,"R");
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
    hist->Fit(fn,"R");
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
  std::string filename = conf_.getUntrackedParameter<std::string>("fileName");
  std::cout << "FileName = " << filename << std::endl;

  //  DQMStore * dbe;

  dbe = edm::Service<DQMStore>().operator->();
  dbe->open(filename);

  //  dbe = edm::Service<DQMStore>().operator->();
  //  dbe->open("/uscms/home/chlebana/DQM_V0001_R000063463__BeamHalo__BeamCommissioning08-PromptReco-v1__RECO.root");

  // print histograms:
  //  dbe->showDirStructure();

  std::vector<MonitorElement*> mes = dbe->getAllContents("");
  std::cout << "found " << mes.size() << " monitoring elements!" << std::endl;

  //   TH1F *bla = fs_->make<TH1F>("bla","bla",256,0,256);
  //   int totF;

  Double_t par[6];
  TF1 *g1    = new TF1("g1","gaus",-50,50);
  TF1 *g2    = new TF1("g2","gaus",-500,500);
  TF1 *dgaus = new TF1("dgaus","gaus(0)+gaus(3)",-500,500);
  TF1 *sgaus = new TF1("sgaus","gaus",-500,500);

  double chindf[6];
  double para[6][6];

  TH1F *hMExy[6];
  TF1  *fitfun[6];
  TF1  *fitfun1[6];
  TF1  *fitfun2[6];
  TH2F *hCaloMEx_LS;
  TH2F *hCaloMEy_LS;
  TH2F *hCaloMExNoHF_LS;
  TH2F *hCaloMEyNoHF_LS;


  // Loop over Monitoring Elements and fill working histograms
  for(std::vector<MonitorElement*>::const_iterator ime = mes.begin(); ime!=mes.end(); ++ime) {
    std::string name = (*ime)->getName();

    //    std::cout << "Name = " << name << std::endl;
    //    if (name == "METTask_CaloMEx") {
    //      std::cout << "Found Name = " << name << " Bins = " << (*ime)->getNbinsX() << std::endl;
    //    }

    if (name == "METTask_CaloMEx")     hMExy[0] = (*ime)->getTH1F();
    if (name == "METTask_CaloMEy")     hMExy[1] = (*ime)->getTH1F();
    if (name == "METTask_CaloMExNoHF") hMExy[2] = (*ime)->getTH1F();
    if (name == "METTask_CaloMEyNoHF") hMExy[3] = (*ime)->getTH1F();

    if (name == "METTask_CaloMEx_LS")     hCaloMEx_LS     = (*ime)->getTH2F();
    if (name == "METTask_CaloMEy_LS")     hCaloMEy_LS     = (*ime)->getTH2F();
    if (name == "METTask_CaloMExNoHF_LS") hCaloMExNoHF_LS = (*ime)->getTH2F();
    if (name == "METTask_CaloMEyNoHF_LS") hCaloMEyNoHF_LS = (*ime)->getTH2F();

  }

  //  std::cout << "Mean = " << hMExy[0]->GetMean() << std::endl;

  for (int i=0;i<4;i++) {
    //    fitd(hMExy[i],dgaus,g1,g2);
    fitfun[i]  = hMExy[i]->GetFunction("dgaus");
    fitfun1[i] = (TF1*)g1->Clone();
    fitfun2[i] = (TF1*)g2->Clone();
  }

  // Slice *_LS histograms
  TH1D *CaloMEx_LS[1000];
  TH1D *CaloMEy_LS[1000];
  TH1D *CaloMExNoHF_LS[1000];
  TH1D *CaloMEyNoHF_LS[1000];
  TF1 *fitfun_CaloMEx_LS[1000];
  TF1 *fitfun_CaloMEy_LS[1000];
  TF1 *fitfun_CaloMExNoHF_LS[1000];
  TF1 *fitfun_CaloMEyNoHF_LS[1000];
  TF1 *fitfun1_CaloMEx_LS[1000];
  TF1 *fitfun1_CaloMEy_LS[1000];
  TF1 *fitfun1_CaloMExNoHF_LS[1000];
  TF1 *fitfun1_CaloMEyNoHF_LS[1000];
  TF1 *fitfun2_CaloMEx_LS[1000];
  TF1 *fitfun2_CaloMEy_LS[1000];
  TF1 *fitfun2_CaloMExNoHF_LS[1000];
  TF1 *fitfun2_CaloMEyNoHF_LS[1000];
  int JetMET_MET[1000];
  int JetMET_MET_All[1000];
  int JetMET_MEx_All[1000];
  int JetMET_MEy_All[1000];
  int JetMET_MET_NoHF[1000];
  int JetMET_MEx_NoHF[1000];
  int JetMET_MEy_NoHF[1000];
  for (int i=0;i<1000;i++){
    JetMET_MET[i]     =-1;
    JetMET_MET_All[i] =-1;
    JetMET_MEx_All[i] =-1;
    JetMET_MEy_All[i] =-1;
    JetMET_MET_NoHF[i]=-1;
    JetMET_MEx_NoHF[i]=-1;
    JetMET_MEy_NoHF[i]=-1;
  }
  char ctitle[100];

  for (int LS=0; LS<1000; LS++){

    // Projection returns a 
    sprintf(ctitle,"CaloMEx_%04d",LS);     CaloMEx_LS[LS]=hCaloMEx_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMEy_%04d",LS);     CaloMEy_LS[LS]=hCaloMEy_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMExNoHF_%04d",LS); CaloMExNoHF_LS[LS]=hCaloMExNoHF_LS->ProjectionX(ctitle,LS+1,LS+1);
    sprintf(ctitle,"CaloMEyNoHF_%04d",LS); CaloMEyNoHF_LS[LS]=hCaloMEyNoHF_LS->ProjectionX(ctitle,LS+1,LS+1);

    if (CaloMEx_LS[LS]->GetSum()>0.) {
      fitdd(CaloMEx_LS[LS],dgaus,g1,g2);
        fitfun_CaloMEx_LS[LS]=CaloMEx_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMEx_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMEx_LS[LS]=(TF1*)g2->Clone();
      fitdd(CaloMEy_LS[LS],dgaus,g1,g2);
        fitfun_CaloMEy_LS[LS]=CaloMEy_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMEy_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMEy_LS[LS]=(TF1*)g2->Clone();
      fitdd(CaloMExNoHF_LS[LS],dgaus,g1,g2);
        fitfun_CaloMExNoHF_LS[LS]=CaloMExNoHF_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMExNoHF_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMExNoHF_LS[LS]=(TF1*)g2->Clone();
      fitdd(CaloMEyNoHF_LS[LS],dgaus,g1,g2);
        fitfun_CaloMEyNoHF_LS[LS]=CaloMEyNoHF_LS[LS]->GetFunction("dgaus");
        fitfun1_CaloMEyNoHF_LS[LS]=(TF1*)g1->Clone();
        fitfun2_CaloMEyNoHF_LS[LS]=(TF1*)g2->Clone();
    }
  }




  //
  //--- Show data certification summary
  std::cout << std::endl;
  printf("| Variable                       |   Reduced chi^2              | Mean               | Width      |\n");
  //
  // Entire run
  for (int i=0;i<4;i++){
    int nmean=1;
    if (fitfun[i]->GetNumberFreeParameters()==3) nmean=4;
    printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
           hMExy[i]->GetName(),
           fitfun[i]->GetChisquare(),fitfun[i]->GetNDF(),
           fitfun[i]->GetChisquare()/fitfun[i]->GetNDF(),
           fitfun[i]->GetParameter(nmean),  fitfun[i]->GetParError(nmean+1),
           fitfun[i]->GetParameter(nmean+1),fitfun[i]->GetParError(nmean+1));
  }
  //
  // Each lumi section
  for (int LS=0; LS<1000; LS++){
    if (CaloMEx_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMEx_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      printf("\n| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEx_LS[LS]->GetName(),
             fitfun_CaloMEx_LS[LS]->GetChisquare(),fitfun_CaloMEx_LS[LS]->GetNDF(),
             fitfun_CaloMEx_LS[LS]->GetChisquare()/fitfun_CaloMEx_LS[LS]->GetNDF(),
             fitfun_CaloMEx_LS[LS]->GetParameter(nmean),  fitfun_CaloMEx_LS[LS]->GetParError(nmean),
             fitfun_CaloMEx_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEx_LS[LS]->GetParError(nmean+1));
      JetMET_MEx_All[LS]=data_certificate(fitfun_CaloMEx_LS[LS]->GetChisquare()/fitfun_CaloMEx_LS[LS]->GetNDF(),
                                          fitfun_CaloMEx_LS[LS]->GetParameter(nmean),
                                          5.,10.);
    }
    if (CaloMEy_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMEy_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEy_LS[LS]->GetName(),
             fitfun_CaloMEy_LS[LS]->GetChisquare(),fitfun_CaloMEy_LS[LS]->GetNDF(),
             fitfun_CaloMEy_LS[LS]->GetChisquare()/fitfun_CaloMEy_LS[LS]->GetNDF(),
             fitfun_CaloMEy_LS[LS]->GetParameter(nmean),  fitfun_CaloMEy_LS[LS]->GetParError(nmean),
             fitfun_CaloMEy_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEy_LS[LS]->GetParError(nmean+1));
      JetMET_MEx_All[LS]=data_certificate(fitfun_CaloMEy_LS[LS]->GetChisquare()/fitfun_CaloMEy_LS[LS]->GetNDF(),
                                          fitfun_CaloMEy_LS[LS]->GetParameter(nmean),
                                          5.,10.);
    }
    if (CaloMExNoHF_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMExNoHF_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMExNoHF_LS[LS]->GetName(),
             fitfun_CaloMExNoHF_LS[LS]->GetChisquare(),fitfun_CaloMExNoHF_LS[LS]->GetNDF(),
             fitfun_CaloMExNoHF_LS[LS]->GetChisquare()/fitfun_CaloMExNoHF_LS[LS]->GetNDF(),
             fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean),  fitfun_CaloMExNoHF_LS[LS]->GetParError(nmean),
             fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean+1),fitfun_CaloMExNoHF_LS[LS]->GetParError(nmean+1))
;
      JetMET_MEx_NoHF[LS]=data_certificate(fitfun_CaloMExNoHF_LS[LS]->GetChisquare()/fitfun_CaloMExNoHF_LS[LS]->GetNDF(),
                                           fitfun_CaloMExNoHF_LS[LS]->GetParameter(nmean),
                                           5.,10.);
    }
    if (CaloMEyNoHF_LS[LS]->GetSum()>0.) {
      int nmean=1;
      if (fitfun_CaloMEyNoHF_LS[LS]->GetNumberFreeParameters()==3) nmean=4;
      printf("| %-30s | %8.3f/%8.3f = %8.3f | %8.3f+-%8.3f | %8.3f+-%8.3f |\n",
             CaloMEyNoHF_LS[LS]->GetName(),
             fitfun_CaloMEyNoHF_LS[LS]->GetChisquare(),fitfun_CaloMEyNoHF_LS[LS]->GetNDF(),
             fitfun_CaloMEyNoHF_LS[LS]->GetChisquare()/fitfun_CaloMEyNoHF_LS[LS]->GetNDF(),
             fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean),  fitfun_CaloMEyNoHF_LS[LS]->GetParError(nmean),
             fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean+1),fitfun_CaloMEyNoHF_LS[LS]->GetParError(nmean+1));
      JetMET_MEy_NoHF[LS]=data_certificate(fitfun_CaloMEyNoHF_LS[LS]->GetChisquare()/fitfun_CaloMEyNoHF_LS[LS]->GetNDF(),
                                           fitfun_CaloMEyNoHF_LS[LS]->GetParameter(nmean),
                                           5.,10.);
    }
  }

  //
  // Data certification format
  std::cout << std::endl;
  int irun=1;
  printf("run, lumi-sec,        tag name, output\n");
  for (int LS=0; LS<1000; LS++){
    JetMET_MET_All[LS] = JetMET_MEx_All[LS] * JetMET_MEy_All[LS];
    JetMET_MET_NoHF[LS]= JetMET_MEx_NoHF[LS]* JetMET_MEy_NoHF[LS];
    JetMET_MET[LS]     = JetMET_MET_All[LS] * JetMET_MET_NoHF[LS];
    if (CaloMEx_LS[LS]->GetSum()>0.) {
      printf("%4d %4d %20s %4d\n",irun,LS,"JetMET_MET",     JetMET_MET[LS]);
      printf("%4d %4d %20s %4d\n",irun,LS,"JetMET_MET_All", JetMET_MET_All[LS]);
      printf("%4d %4d %20s %4d\n",irun,LS,"JetMET_MET_NoHF",JetMET_MET_NoHF[LS]);
    }
  }

  std::cout << std::endl;







//     if(name.find(tagname)>=name.size())
//       continue;
//     totF++;
//     std::cout << "hm found " << name << std::endl;
//     float filled=0;
//     float tot=0;
//     for(int ix=0; ix<=(*ime)->getNbinsX(); ++ix){
//       for(int iy=0; iy<=(*ime)->getNbinsY(); ++iy){
//      tot++;
//      if((*ime)->getBinContent(ix,iy)>0){
//        filled++;
//        std::cout << " " << (*ime)->getBinContent(ix,iy);
//        bla->Fill((*ime)->getBinContent(ix,iy));
//      }
//       }
//       std::cout << std::endl;
//     }
//     std::cout << name  << " " << filled/tot << std::endl;
//   }
//   std::cout << "tot " << totF << std::endl;


}

// ------------ method called once each job just after ending the event loop  ------------
void 
DataCertificationJetMET::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DataCertificationJetMET);
