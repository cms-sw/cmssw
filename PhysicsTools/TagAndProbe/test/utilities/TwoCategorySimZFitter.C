/*******************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: Presently in the user code
 *
 *
 * Authors:
 *
 *   Kalanand Mishra, Fermilab - kalanand@fnal.gov
 *
 * Description:
 *   A standalone macro to perform single bin simultaneos fit of 
 *   dilepton mass under Z peak. 
 *   Works with  unbinned data. 
 *
 * Implementation details:
 *  Uses RooFit classes.
 *
 * History:
 *   
 *
 ********************************************************************/

//// Following are the 2 irreducible free parameters of the fit:
////        Z cross section and single electron efficiency.
////  Additionally, the following 2 nuisance parameters are floating:
////        nBkgTF, bkgGamma.


// ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include <TGraph.h>
#include "tdrstyle.C"

using namespace RooFit;

void makeSignalPdf();
void makeBkgPdf();

// The signal & background Pdf 
RooRealVar *rooMass_;
RooAbsPdf* signalShapePdfTT_;
RooAbsPdf* signalShapePdfTF_;
RooAbsPdf *bkgShapePdfTF_;


TFile* Zeelineshape_file;
TCanvas *c;

// Acceptance for Gen-->SC 
const float A_BB= 0.2257;
const float A_BE= 0.1612;
const float A_EE= 0.0476;

// Specify the electron selection: "WP95" or "WP80"
const char* selection = "WP95";


// Integrated luminosity
const double intLumi = 2.88;


// In case we want to fix the nuisance parameters ...
const bool FIX_NUIS_PARS=false;




void TwoCategorySimZFitter()
{
  // The fit variable - lepton invariant mass
  rooMass_ = new RooRealVar("mass","m_{ee}",60.0, 120.0, "GeV/c^{2}");
  rooMass_->setBins(120.0);
  RooRealVar Mass = *rooMass_;

  // Make the category variable that defines the two fits,
  // namely whether the probe passes or fails the eff criteria.
  RooCategory sample("sample","") ;
  sample.defineType("pass", 1) ;
  sample.defineType("fail", 0) ; 


  gROOT->cd();
  char inFile[50];
  sprintf(inFile, "ZeeEvents_%s.root", selection);
  TFile fin(inFile, "read");
  TTree* treeTT = (TTree*) fin.Get("tree");
  sprintf(inFile, "ZeeEvents_%s_TF.root", selection);
  TFile fin2(inFile, "read");
  TTree* treeTF = (TTree*) fin2.Get("tree");


  ///////// convert Histograms into RooDataHists
  RooDataSet* data_TT = new RooDataSet("data_TT","data_TT", treeTT, Mass);
  RooDataSet* data_TF = new RooDataSet("data_TF_BB","data_TF",treeTF, Mass);

  RooDataSet* data = new RooDataSet( "data","data",
				       RooArgList(Mass),Index(sample),
				       Import("pass",*data_TT),
				       Import("fail",*data_TF) ); 

  data->get()->Print();
  cout << "Made dataset" << endl;




  // ********** Construct signal & bkg shape PDF ********** //
  makeSignalPdf();
  cout << "Made signal pdf" << endl;
  makeBkgPdf();
  cout << "Made bkg pdf" << endl;

  // Now supply integrated luminosity in inverse picobarn
  RooRealVar lumi("lumi","lumi", intLumi);


  // Now define Z production cross section variable (in pb) 
  RooRealVar xsec("xsec","xsec", 912.187, 200.0, 2000.0);


  // Define efficiency variable: single bin efficiency, ignore trigger for now
  RooRealVar eff("eff","eff", 0.93, 0.0, 1.0);


  // Now define acceptance variables-->we get these numbers from MC  
  RooRealVar acc("acc","acc", A_BB+A_BE+A_EE);



  // Define background yield variables: they are not related to each other  
  float numBkgHighPurity=11.8;
  if(!strcmp(selection,"WP80")) numBkgHighPurity=3.0; 
  RooRealVar nBkgTT("nBkgTT","nBkgTT", numBkgHighPurity);
  RooRealVar nBkgTF("nBkgTF","nBkgTF", 58.0,     0.0, 500.);
  if(!strcmp(selection,"WP80")) {
     nBkgTF.setVal(0.0);
     nBkgTF.setConstant(true);
  }
   if(FIX_NUIS_PARS) nBkgTF.setConstant(true);

  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////
 //  Define signal yield variables.  
  // They are linked together by the total cross section:  e.g. 
  //          Nbb = sigma*L*Abb*effB

  const char* formula = 0;
  RooArgList* args;
  formula="lumi*xsec*acc*eff*eff+nBkgTT";
  args = new RooArgList(lumi,xsec,acc,eff,nBkgTT);
  RooFormulaVar nSigTT("nSigTT", formula, *args);
  delete args;

  formula="lumi*xsec*acc*eff*(1.0-eff)";
  args = new RooArgList(lumi,xsec,acc,eff);
  RooFormulaVar nSigTF("nSigTF",formula, *args);
  delete args;

  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

   RooArgList componentsTF(*signalShapePdfTF_,*bkgShapePdfTF_);
   RooArgList yieldsTF(nSigTF, nBkgTF);	  
   RooExtendPdf pdfTT("pdfTT","extended sum pdf", *signalShapePdfTT_, nSigTT);
   RooAddPdf pdfTF("pdfTF","extended sum pdf",componentsTF, yieldsTF);
 

   // The total simultaneous fit ...
   RooSimultaneous totalPdf("totalPdf","totalPdf", sample);
   totalPdf.addPdf(pdfTT,"pass");
   totalPdf.Print();
   totalPdf.addPdf(pdfTF,"fail");
   totalPdf.Print();


  // ********* Do the Actual Fit ********** //  
   RooFitResult *fitResult = totalPdf.fitTo(*data, Save(true), 
   RooFit::Extended(true), 
   RooFit::Minos(true), 
   PrintEvalErrors(-1),
   Warnings(false) 
      );
  fitResult->Print("v");

  // ********** Make and save Canvas for the plots ********** //
  gROOT->ProcessLine(".L ~/tdrstyle.C");
  setTDRStyle();
  tdrStyle->SetErrorX(0.5);
  tdrStyle->SetPadLeftMargin(0.19);
  tdrStyle->SetPadRightMargin(0.10);
  tdrStyle->SetPadBottomMargin(0.15);
  tdrStyle->SetLegendBorderSize(0);
  tdrStyle->SetTitleYOffset(1.5);
  RooAbsData::ErrorType errorType = RooAbsData::SumW2;


  TString cname = Form("Zmass_TT_%dnb", (int)(1000*intLumi) );
  c = new TCanvas(cname,cname,500,500);
  RooPlot* frame1 = Mass.frame(60., 120., 60);
  data_TT->plotOn(frame1,RooFit::DataError(errorType));
  pdfTT.plotOn(frame1,ProjWData(*data_TT));
  frame1->SetMinimum(0);
  frame1->Draw("e0");
  TPaveText *plotlabel = new TPaveText(0.23,0.87,0.43,0.92,"NDC");
   plotlabel->SetTextColor(kBlack);
   plotlabel->SetFillColor(kWhite);
   plotlabel->SetBorderSize(0);
   plotlabel->SetTextAlign(12);
   plotlabel->SetTextSize(0.03);
   plotlabel->AddText("CMS Preliminary 2010");
  TPaveText *plotlabel2 = new TPaveText(0.23,0.82,0.43,0.87,"NDC");
   plotlabel2->SetTextColor(kBlack);
   plotlabel2->SetFillColor(kWhite);
   plotlabel2->SetBorderSize(0);
   plotlabel2->SetTextAlign(12);
   plotlabel2->SetTextSize(0.03);
   plotlabel2->AddText("#sqrt{s} = 7 TeV");
  TPaveText *plotlabel3 = new TPaveText(0.23,0.75,0.43,0.80,"NDC");
   plotlabel3->SetTextColor(kBlack);
   plotlabel3->SetFillColor(kWhite);
   plotlabel3->SetBorderSize(0);
   plotlabel3->SetTextAlign(12);
   plotlabel3->SetTextSize(0.03);
  char temp[100];
  sprintf(temp, "%.1f", intLumi);
  plotlabel3->AddText((string("#int#font[12]{L}dt = ") + 
  temp + string(" pb^{ -1}")).c_str());
  TPaveText *plotlabel4 = new TPaveText(0.6,0.87,0.8,0.92,"NDC");
   plotlabel4->SetTextColor(kBlack);
   plotlabel4->SetFillColor(kWhite);
   plotlabel4->SetBorderSize(0);
   plotlabel4->SetTextAlign(12);
   plotlabel4->SetTextSize(0.03);
   double nsig = nSigTT.getVal();
   double nsigerr = nSigTT.getPropagatedError(*fitResult) ; 
   sprintf(temp, "Signal = %.1f #pm %.1f", nsig, nsigerr);
   plotlabel4->AddText(temp);
   TPaveText *plotlabel5 = new TPaveText(0.6,0.82,0.8,0.87,"NDC");
   plotlabel5->SetTextColor(kBlack);
   plotlabel5->SetFillColor(kWhite);
   plotlabel5->SetBorderSize(0);
   plotlabel5->SetTextAlign(12);
   plotlabel5->SetTextSize(0.03);
   sprintf(temp, "#epsilon = %.3f #pm %.3f", eff.getVal(), eff.getError() );
   plotlabel5->AddText(temp);
  TPaveText *plotlabel7 = new TPaveText(0.6,0.77,0.8,0.82,"NDC");
   plotlabel7->SetTextColor(kBlack);
   plotlabel7->SetFillColor(kWhite);
   plotlabel7->SetBorderSize(0);
   plotlabel7->SetTextAlign(12);
   plotlabel7->SetTextSize(0.03);
   sprintf(temp, "#sigma = %.1f #pm %.1f pb", xsec.getVal(), xsec.getError());
   plotlabel7->AddText(temp);
  plotlabel->Draw();
  plotlabel2->Draw();
  plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel7->Draw();
//   c->SaveAs( cname + TString(".eps"));
//   c->SaveAs( cname + TString(".gif"));
//   c->SaveAs( cname + TString(".root"));
//   c->SaveAs( cname + TString(".png"));
//   c->SaveAs( cname + TString(".C"));


  cname = Form("Zmass_TF%dnb", (int)(1000*intLumi) );
  c = new TCanvas(cname,cname,500,500);
  RooPlot* frame2 = Mass.frame(60., 120., 12);
  data_TF->plotOn(frame2,RooFit::DataError(errorType));
  pdfTF.plotOn(frame2,ProjWData(*data_TF),
  Components(*signalShapePdfTF_),DrawOption("LF"),FillStyle(1001),FillColor(kOrange),VLines());
  pdfTF.plotOn(frame2,ProjWData(*data_TF));
  pdfTF.plotOn(frame2,ProjWData(*data_TF),
  Components(*bkgShapePdfTF_),LineColor(kRed));
  data_TF->plotOn(frame2,RooFit::DataError(errorType));
  frame2->SetMinimum(0);
  frame2->Draw("e0");
  frame2->GetYaxis()->SetNdivisions(505);
  plotlabel = new TPaveText(0.23,0.87,0.43,0.92,"NDC");
   plotlabel->SetTextColor(kBlack);
   plotlabel->SetFillColor(kWhite);
   plotlabel->SetBorderSize(0);
   plotlabel->SetTextAlign(12);
   plotlabel->SetTextSize(0.03);
   plotlabel->AddText("CMS Preliminary 2010");
  plotlabel2 = new TPaveText(0.23,0.82,0.43,0.87,"NDC");
   plotlabel2->SetTextColor(kBlack);
   plotlabel2->SetFillColor(kWhite);
   plotlabel2->SetBorderSize(0);
   plotlabel2->SetTextAlign(12);
   plotlabel2->SetTextSize(0.03);
   plotlabel2->AddText("#sqrt{s} = 7 TeV");
  plotlabel3 = new TPaveText(0.23,0.75,0.43,0.80,"NDC");
   plotlabel3->SetTextColor(kBlack);
   plotlabel3->SetFillColor(kWhite);
   plotlabel3->SetBorderSize(0);
   plotlabel3->SetTextAlign(12);
   plotlabel3->SetTextSize(0.03);
  char temp2[100];
  sprintf(temp2, "%.1f", intLumi);
  plotlabel3->AddText((string("#int#font[12]{L}dt = ") + 
  temp2 + string(" pb^{ -1}")).c_str());
  plotlabel4 = new TPaveText(0.6,0.87,0.8,0.92,"NDC");
   plotlabel4->SetTextColor(kBlack);
   plotlabel4->SetFillColor(kWhite);
   plotlabel4->SetBorderSize(0);
   plotlabel4->SetTextAlign(12);
   plotlabel4->SetTextSize(0.03);
   nsig = nSigTF.getVal();
   nsigerr = nSigTF.getPropagatedError(*fitResult) ; 
   sprintf(temp2, "Signal = %.2f #pm %.2f", nsig, nsigerr);
   plotlabel4->AddText(temp2);
  plotlabel5 = new TPaveText(0.6,0.82,0.8,0.87,"NDC");
   plotlabel5->SetTextColor(kBlack);
   plotlabel5->SetFillColor(kWhite);
   plotlabel5->SetBorderSize(0);
   plotlabel5->SetTextAlign(12);
   plotlabel5->SetTextSize(0.03);
   sprintf(temp2, "Bkg = %.2f #pm %.2f", nBkgTF.getVal(), nBkgTF.getError());
   plotlabel5->AddText(temp2);
  TPaveText* plotlabel6 = new TPaveText(0.6,0.77,0.8,0.82,"NDC");
   plotlabel6->SetTextColor(kBlack);
   plotlabel6->SetFillColor(kWhite);
   plotlabel6->SetBorderSize(0);
   plotlabel6->SetTextAlign(12);
   plotlabel6->SetTextSize(0.03);
   sprintf(temp2, "#epsilon = %.3f #pm %.3f", eff.getVal(), eff.getError() );
   plotlabel6->AddText(temp2);
  plotlabel->Draw();
  plotlabel2->Draw();
  plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();

//   c->SaveAs( cname + TString(".eps"));
//   c->SaveAs( cname + TString(".gif"));
//   c->SaveAs( cname + TString(".root"));
//   c->SaveAs( cname + TString(".png"));
//   c->SaveAs( cname + TString(".C"));


  //    if(data) delete data;
  //    if(c) delete c;
}





// // ***** Function to return the signal Pdf *** //
 void makeSignalPdf() {

 // Tag+Tag selection pdf
  Zeelineshape_file =  new TFile("../Zlineshapes.root", "READ");

  // Tag+Tag selection pdf
  TH1* histbbpass = (TH1D*) Zeelineshape_file->Get("pass_BB");
  TH1* histebpass = (TH1D*) Zeelineshape_file->Get("pass_BE");
  TH1* histeepass = (TH1D*) Zeelineshape_file->Get("pass_EE");

  TH1D* th1_TT = (TH1D*) histbbpass->Clone("th1_TT");
  th1_TT->Add(histebpass);
  th1_TT->Add(histeepass);

  RooRealVar* zero  = new RooRealVar("zero","", 0.0);
  RooDataHist* rdh_TT = new RooDataHist("rdh_TT","", *rooMass_, th1_TT);
  RooAbsPdf* signalModelTT_ = new RooHistPdf("signalModelTT", "", *rooMass_, *rdh_TT);
  RooRealVar* resoTT_  = new RooRealVar("resoTT","resoTT",1.88, 0.0, 5.);
  if(FIX_NUIS_PARS) resoTT_->setConstant(true);
  RooGaussModel* resModelTT_ = new RooGaussModel("resModelTT","", 
                *rooMass_, *zero, *resoTT_);
  signalShapePdfTT_  = new RooFFTConvPdf("sigModel","final signal shape", 
                       *rooMass_, *signalModelTT_, *resModelTT_);


 // Tag+Fail selection pdf
  TH1* histbb = (TH1D*) Zeelineshape_file->Get("fail_BB");
  TH1* histeb = (TH1D*) Zeelineshape_file->Get("fail_BE");
  TH1* histee = (TH1D*) Zeelineshape_file->Get("fail_EE");

  TH1D* th1_TF = (TH1D*) histbb->Clone("th1_TF");
  th1_TF->Add(histeb);
  th1_TF->Add(histee);
  RooDataHist* rdh_TF_ = new RooDataHist("rdh_TF_BB","", *rooMass_, th1_TF);
  RooAbsPdf* signalModelTF_ = new RooHistPdf("signalModelTF", 
  "",*rooMass_,*rdh_TF_);
  RooRealVar* resoTF_  = new RooRealVar("resoTF","resoTF", 2.76762, 0.0, 10.);
  if(FIX_NUIS_PARS) resoTF_->setConstant(true);
  RooGaussModel* resModelTF_ = new RooGaussModel("resModelTF",
  "gaussian resolution model", *rooMass_, *zero, *resoTF_);
  signalShapePdfTF_  = new RooFFTConvPdf("signalShapePdfTF","", 
                       *rooMass_, *signalModelTF_, *resModelTF_);
}




// ***** Function to return the background Pdf **** //
void makeBkgPdf()
{  
  // Background PDF variables
   RooRealVar* bkgGammaFailTF_ = new RooRealVar("bkgGammaFailTF",
   "",-0.028394, -10., 10.);
   if(FIX_NUIS_PARS) bkgGammaFailTF_->setConstant(true);
   bkgShapePdfTF_ = new RooExponential("bkgShapePdfTF",
   "",*rooMass_, *bkgGammaFailTF_);
}









