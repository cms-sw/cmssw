//================================================================================================
//
//
//________________________________________________________________________________________________

#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TROOT.h>       // access to gROOT, entry point to ROOT system
#include <TSystem.h>     // interface to OS
#include <TStyle.h>      // class to handle ROOT plotting style
#include <TCanvas.h>     // class for drawing
#include <TBenchmark.h>  // class to track macro running statistics
#include <iostream>      // standard I/O

// RooFit headers
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooCategory.h"
#include "RooArgList.h"
#include "RooDataHist.h"
#include "RooFormulaVar.h"
#include "RooHistPdf.h"
#include "RooGenericPdf.h"
#include "RooAddPdf.h"
#include "RooSimultaneous.h"
#include "RooGaussian.h"
#include "RooNLLVar.h"
#include "RooConstVar.h"
#include "RooFitResult.h"
#include "RooExponential.h"
#include "RooFFTConvPdf.h"

#include "TFile.h"
#include "TH1D.h"
#include "TPaveText.h"
#include "RooPlot.h"

#endif

#define LUMINOSITY 2.88  //(in pb^-1)
#define NBINSPASS 24
#define NBINSFAIL 24

using namespace RooFit;

double ErrorInProduct(double x, double errx, double y, double erry, double corr) {
  double xFrErr = errx / x;
  double yFrErr = erry / y;
  return sqrt(pow(xFrErr, 2) + pow(yFrErr, 2) + 2.0 * corr * xFrErr * yFrErr) * x * y;
}

//=== MAIN MACRO =================================================================================================

void performFit(string inputDir,
                string OSInputDataFilename,
                string SSInputDataFilename,
                string OSSignalTemplateHistName,
                string SSSignalTemplateHistName) {
  gBenchmark->Start("fitZCat");

  //--------------------------------------------------------------------------------------------------------------
  // Settings
  //==============================================================================================================

  const Double_t mlow = 60;
  const Double_t mhigh = 120;
  const Int_t nbins = 20;

  // Which fit parameters to set constant
  // MuTrk
  Bool_t cBgA1MuTrk = kFALSE;
  Bool_t cBgA2MuTrk = kTRUE;
  Bool_t cBgAlphaMuTrk = kFALSE;
  // MuMuNoIso
  Bool_t cBgA1MuMuNoIso = kFALSE;
  Bool_t cBgAlphaMuMuNoIso = kFALSE;
  // MuSta
  Bool_t cBgA1MuSta = kFALSE;
  Bool_t cBgAlphaMuSta = kFALSE;

  //--------------------------------------------------------------------------------------------------------------
  // Main analysis code
  //==============================================================================================================
  RooRealVar mass("mass", "mass", mlow, mhigh);
  mass.setBins(nbins);

  //
  // Prepare data for the fits
  //
  char datname[100];
  RooDataSet* dataOS = RooDataSet::read((inputDir + OSInputDataFilename).c_str(), RooArgList(mass));
  RooDataSet* dataSS = RooDataSet::read((inputDir + SSInputDataFilename).c_str(), RooArgList(mass));

  //Define categories
  RooCategory sample("sample", "");
  sample.defineType("OS", 1);
  sample.defineType("SS", 2);

  RooDataSet* dataCombined = new RooDataSet(
      "dataCombined", "dataCombined", RooArgList(mass), Index(sample), Import("OS", *dataOS), Import("SS", *dataSS));

  //*********************************************************************************************
  //Define Free Parameters
  //*********************************************************************************************

  RooRealVar* ParNumSignalOS = new RooRealVar("ParNumSignalOS", "ParNumSignalOS", 1000.0, 0.0, 10000000.0);
  RooRealVar* ParNumSignalSS = new RooRealVar("ParNumSignalSS", "ParNumSignalSS", 100.0, 0.0, 10000000.0);
  RooRealVar* ParNumBkgOS = new RooRealVar("ParNumBkgOS", "ParNumBkgOS", 10.0, 0.0, 10000000.0);
  RooRealVar* ParNumBkgSS = new RooRealVar("ParNumBkgSS", "ParNumBkgSS", 1.0, 0.0, 10000000.0);

  RooRealVar* ParOSBackgroundExpCoefficient = new RooRealVar("ParOSBackgroundExpCoefficient",
                                                             "ParOSBackgroundExpCoefficient",
                                                             -0.01,
                                                             -1.0,
                                                             1.0);  //ParOSBackgroundExpCoefficient.setConstant(kTRUE);
  RooRealVar* ParSSBackgroundExpCoefficient = new RooRealVar("ParSSBackgroundExpCoefficient",
                                                             "ParSSBackgroundExpCoefficient",
                                                             -0.01,
                                                             -1.0,
                                                             1.0);  //ParSSBackgroundExpCoefficient.setConstant(kTRUE);
  RooRealVar* ParOSSignalMassShift = new RooRealVar("ParOSSignalMassShift", "ParOSSignalMassShift", 0.0, -10.0, 10.0);
  ParOSSignalMassShift->setConstant(kTRUE);
  RooRealVar* ParSSSignalMassShift = new RooRealVar("ParSSSignalMassShift", "ParSSSignalMassShift", 0.0, -10.0, 10.0);
  ParSSSignalMassShift->setConstant(kTRUE);
  RooRealVar* ParOSSignalResolution = new RooRealVar(
      "ParOSSignalResolution", "ParOSSignalResolution", 1.0, 0.0, 10.0);  //ParOSSignalResolution.setConstant(kTRUE);
  RooRealVar* ParSSSignalResolution = new RooRealVar(
      "ParSSSignalResolution", "ParSSSignalResolution", 1.0, 0.0, 10.0);  //ParSSSignalResolution.setConstant(kTRUE);

  //*********************************************************************************************
  //
  //Load Signal PDFs
  //
  //*********************************************************************************************
  //Load histogram templates
  TFile* Zeelineshape_file = new TFile("ZMassLineshape_SSOS.root", "READ");
  TH1* histTemplateOS = (TH1D*)Zeelineshape_file->Get(OSSignalTemplateHistName.c_str());
  TH1* histTemplateSS = (TH1D*)Zeelineshape_file->Get(SSSignalTemplateHistName.c_str());

  //Introduce mass shift coordinate transformation
  RooFormulaVar OSShiftedMass("OSShiftedMass", "@0-@1", RooArgList(mass, *ParOSSignalMassShift));
  RooFormulaVar SSShiftedMass("SSShiftedMass", "@0-@1", RooArgList(mass, *ParSSSignalMassShift));

  RooDataHist* dataHistOS = new RooDataHist("dataHistOS", "dataHistOS", RooArgSet(mass), histTemplateOS);
  RooDataHist* dataHistSS = new RooDataHist("dataHistSS", "dataHistSS", RooArgSet(mass), histTemplateSS);

  RooGaussian* OSSignalResolution =
      new RooGaussian("OSSignalResolution", "OSSignalResolution", mass, *ParOSSignalMassShift, *ParOSSignalResolution);
  RooGaussian* SSSignalResolution =
      new RooGaussian("SSSignalResolution", "SSSignalResolution", mass, *ParSSSignalMassShift, *ParSSSignalResolution);

  RooHistPdf* OSSignalShapeTemplatePdf =
      new RooHistPdf("OSSignalShapeTemplatePdf", "OSSignalShapeTemplatePdf", RooArgSet(mass), *dataHistOS, 1);
  RooHistPdf* SSSignalShapeTemplatePdf =
      new RooHistPdf("SSSignalShapeTemplatePdf", "SSSignalShapeTemplatePdf", RooArgSet(mass), *dataHistSS, 1);

  RooFFTConvPdf* signalShapeOSPdf = new RooFFTConvPdf(
      "signalShapeOSPdf", "signalShapeOSPdf", mass, *OSSignalShapeTemplatePdf, *OSSignalResolution, 2);
  RooFFTConvPdf* signalShapeSSPdf = new RooFFTConvPdf(
      "signalShapeSSPdf", "signalShapeSSPdf", mass, *SSSignalShapeTemplatePdf, *SSSignalResolution, 2);

  //*********************************************************************************************
  //
  // Create Background PDFs
  //
  //*********************************************************************************************
  RooExponential* bkgOSPdf = new RooExponential("bkgOSPdf", "bkgOSPdf", mass, *ParOSBackgroundExpCoefficient);
  RooExponential* bkgSSPdf = new RooExponential("bkgSSPdf", "bkgSSPdf", mass, *ParSSBackgroundExpCoefficient);

  //*********************************************************************************************
  //
  // Create Total PDFs
  //
  //*********************************************************************************************
  RooAddPdf pdfOS(
      "pdfOS", "pdfOS", RooArgList(*signalShapeOSPdf, *bkgOSPdf), RooArgList(*ParNumSignalOS, *ParNumBkgOS));
  RooAddPdf pdfSS(
      "pdfSS", "pdfSS", RooArgList(*signalShapeSSPdf, *bkgSSPdf), RooArgList(*ParNumSignalSS, *ParNumBkgSS));

  // PDF for simultaneous fit
  RooSimultaneous pdfTotal("pdfTotal", "pdfTotal", sample);
  pdfTotal.addPdf(pdfOS, "pdfOS");
  pdfTotal.addPdf(pdfSS, "pdfSS");
  pdfTotal.Print();

  //
  // Define likelihood, add constraints, and run the fit
  //
  RooFitResult* fitResult =
      pdfTotal.fitTo(*dataCombined, RooFit::Save(true), RooFit::Extended(true), RooFit::PrintLevel(-1));
  fitResult->Print("v");

  double nSignalOS = ParNumSignalOS->getVal();
  double nSignalSS = ParNumSignalSS->getVal();
  double nBkgOS = ParNumBkgOS->getVal();
  double nBkgSS = ParNumBkgSS->getVal();

  printf("\nFit results:\n");
  cout << "Signal OS: " << nSignalOS << endl;
  cout << "Bkg OS: " << nBkgOS << endl;
  cout << "Signal SS: " << nSignalSS << endl;
  cout << "Bkg SS: " << nBkgSS << endl;

  //--------------------------------------------------------------------------------------------------------------
  // Make plots
  //==============================================================================================================
  RooAbsData::ErrorType errorType = RooAbsData::Poisson;

  TString cname = TString("fit_OS");
  TCanvas* c = new TCanvas(cname, cname, 800, 600);
  RooPlot* frame1 = mass.frame();
  frame1->SetMinimum(0);
  dataOS->plotOn(frame1, RooFit::DataError(errorType));
  pdfOS.plotOn(frame1, RooFit::ProjWData(*dataOS), RooFit::Components(*bkgOSPdf), RooFit::LineColor(kRed));
  pdfOS.plotOn(frame1, RooFit::ProjWData(*dataOS));
  frame1->Draw("e0");

  TPaveText* plotlabel = new TPaveText(0.23, 0.87, 0.43, 0.92, "NDC");
  plotlabel->SetTextColor(kBlack);
  plotlabel->SetFillColor(kWhite);
  plotlabel->SetBorderSize(0);
  plotlabel->SetTextAlign(12);
  plotlabel->SetTextSize(0.03);
  plotlabel->AddText("CMS Preliminary 2010");
  TPaveText* plotlabel2 = new TPaveText(0.23, 0.82, 0.43, 0.87, "NDC");
  plotlabel2->SetTextColor(kBlack);
  plotlabel2->SetFillColor(kWhite);
  plotlabel2->SetBorderSize(0);
  plotlabel2->SetTextAlign(12);
  plotlabel2->SetTextSize(0.03);
  plotlabel2->AddText("#sqrt{s} = 7 TeV");
  TPaveText* plotlabel3 = new TPaveText(0.23, 0.75, 0.43, 0.80, "NDC");
  plotlabel3->SetTextColor(kBlack);
  plotlabel3->SetFillColor(kWhite);
  plotlabel3->SetBorderSize(0);
  plotlabel3->SetTextAlign(12);
  plotlabel3->SetTextSize(0.03);
  char temp[100];
  sprintf(temp, "%.4f", LUMINOSITY);
  plotlabel3->AddText((string("#int#font[12]{L}dt = ") + temp + string(" pb^{ -1}")).c_str());
  TPaveText* plotlabel4 = new TPaveText(0.6, 0.82, 0.8, 0.87, "NDC");
  plotlabel4->SetTextColor(kBlack);
  plotlabel4->SetFillColor(kWhite);
  plotlabel4->SetBorderSize(0);
  plotlabel4->SetTextAlign(12);
  plotlabel4->SetTextSize(0.03);
  sprintf(temp, "Signal = %.2f #pm %.2f", ParNumSignalOS->getVal(), ParNumSignalOS->getError());
  plotlabel4->AddText(temp);
  TPaveText* plotlabel5 = new TPaveText(0.6, 0.77, 0.8, 0.82, "NDC");
  plotlabel5->SetTextColor(kBlack);
  plotlabel5->SetFillColor(kWhite);
  plotlabel5->SetBorderSize(0);
  plotlabel5->SetTextAlign(12);
  plotlabel5->SetTextSize(0.03);
  sprintf(temp, "Bkg = %.2f #pm %.2f", ParNumBkgOS->getVal(), ParNumBkgOS->getError());
  plotlabel5->AddText(temp);
  plotlabel4->Draw();
  plotlabel5->Draw();

  //   c->SaveAs( cname + TString(".eps"));
  c->SaveAs(cname + TString(".gif"));
  delete c;

  cname = TString("fit_SS");
  TCanvas* c2 = new TCanvas(cname, cname, 500, 500);
  RooPlot* frame2 = mass.frame();
  frame2->SetMinimum(0);
  dataSS->plotOn(frame2, RooFit::DataError(errorType));
  pdfSS.plotOn(frame2, RooFit::ProjWData(*dataSS), RooFit::Components(*bkgSSPdf), RooFit::LineColor(kRed));
  pdfSS.plotOn(frame2, RooFit::ProjWData(*dataSS));
  frame2->Draw("e0");

  plotlabel = new TPaveText(0.23, 0.87, 0.43, 0.92, "NDC");
  plotlabel->SetTextColor(kBlack);
  plotlabel->SetFillColor(kWhite);
  plotlabel->SetBorderSize(0);
  plotlabel->SetTextAlign(12);
  plotlabel->SetTextSize(0.03);
  plotlabel->AddText("CMS Preliminary 2010");
  plotlabel2 = new TPaveText(0.23, 0.82, 0.43, 0.87, "NDC");
  plotlabel2->SetTextColor(kBlack);
  plotlabel2->SetFillColor(kWhite);
  plotlabel2->SetBorderSize(0);
  plotlabel2->SetTextAlign(12);
  plotlabel2->SetTextSize(0.03);
  plotlabel2->AddText("#sqrt{s} = 7 TeV");
  plotlabel3 = new TPaveText(0.23, 0.75, 0.43, 0.80, "NDC");
  plotlabel3->SetTextColor(kBlack);
  plotlabel3->SetFillColor(kWhite);
  plotlabel3->SetBorderSize(0);
  plotlabel3->SetTextAlign(12);
  plotlabel3->SetTextSize(0.03);
  sprintf(temp, "%.4f", LUMINOSITY);
  plotlabel3->AddText((string("#int#font[12]{L}dt = ") + temp + string(" pb^{ -1}")).c_str());
  plotlabel4 = new TPaveText(0.6, 0.82, 0.8, 0.87, "NDC");
  plotlabel4->SetTextColor(kBlack);
  plotlabel4->SetFillColor(kWhite);
  plotlabel4->SetBorderSize(0);
  plotlabel4->SetTextAlign(12);
  plotlabel4->SetTextSize(0.03);
  sprintf(temp, "Signal = %.2f #pm %.2f", ParNumSignalSS->getVal(), ParNumSignalSS->getError());
  plotlabel4->AddText(temp);
  plotlabel5 = new TPaveText(0.6, 0.77, 0.8, 0.82, "NDC");
  plotlabel5->SetTextColor(kBlack);
  plotlabel5->SetFillColor(kWhite);
  plotlabel5->SetBorderSize(0);
  plotlabel5->SetTextAlign(12);
  plotlabel5->SetTextSize(0.03);
  sprintf(temp, "Bkg = %.2f #pm %.2f", ParNumBkgSS->getVal(), ParNumBkgSS->getError());
  plotlabel5->AddText(temp);

  //   plotlabel->Draw();
  //   plotlabel2->Draw();
  //   plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();

  c2->SaveAs(cname + TString(".eps"));
  c2->SaveAs(cname + TString(".gif"));
  c2->SaveAs(cname + TString(".root"));
  delete c2;

  gBenchmark->Show("fitZCat");
}

void ZeeBkgFit() {
  performFit("MitFitter/inputs/BkgFit/",
             "M_VBTF95PlusVBTF95_OS",
             "M_VBTF95PlusVBTF95_SS",
             "Mass_WP95WP95_OppositeSign",
             "Mass_WP95WP95_SameSign");
}
