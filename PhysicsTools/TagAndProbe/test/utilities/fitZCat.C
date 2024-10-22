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
#include "RooConstVar.h"
#include "RooFitResult.h"
#include "RooExponential.h"

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
                string PassInputDataFilename,
                string FailInputDataFilename,
                string PassSignalTemplateHistName,
                string FailSignalTemplateHistName) {
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
  //Define categories
  RooCategory sample("sample", "");
  sample.defineType("Pass", 1);
  sample.defineType("Fail", 2);

  char datname[100];
  RooDataSet* dataPass = RooDataSet::read((inputDir + PassInputDataFilename).c_str(), RooArgList(mass));
  RooDataSet* dataFail = RooDataSet::read((inputDir + FailInputDataFilename).c_str(), RooArgList(mass));

  RooDataSet* dataCombined = new RooDataSet("dataCombined",
                                            "dataCombined",
                                            RooArgList(mass),
                                            Index(sample),
                                            Import("Pass", *dataPass),
                                            Import("Fail", *dataFail));

  //*********************************************************************************************
  //Define Free Parameters
  //*********************************************************************************************
  RooRealVar* ParEfficiency = new RooRealVar("ParEfficiency", "ParEfficiency", 0.9, 0.0, 1.0);
  RooRealVar* ParNumSignal = new RooRealVar("ParNumSignal", "ParNumSignal", 4000.0, 0.0, 100000.0);
  RooRealVar* ParNumBkgPass = new RooRealVar("ParNumBkgPass", "ParNumBkgPass", 1000.0, 0.0, 10000000.0);
  RooRealVar* ParNumBkgFail = new RooRealVar("ParNumBkgFail", "ParNumBkgFail", 1000.0, 0.0, 10000000.0);

  RooRealVar* ParPassBackgroundExpCoefficient =
      new RooRealVar("ParPassBackgroundExpCoefficient",
                     "ParPassBackgroundExpCoefficient",
                     -0.2,
                     -1.0,
                     0.0);  //ParPassBackgroundExpCoefficient.setConstant(kTRUE);
  RooRealVar* ParFailBackgroundExpCoefficient =
      new RooRealVar("ParFailBackgroundExpCoefficient",
                     "ParFailBackgroundExpCoefficient",
                     -0.2,
                     -1.0,
                     0.0);  //ParFailBackgroundExpCoefficient.setConstant(kTRUE);
  RooRealVar* ParPassSignalMassShift = new RooRealVar(
      "ParPassSignalMassShift", "ParPassSignalMassShift", -10.0, 10.0);  //ParPassSignalMassShift.setConstant(kTRUE);
  RooRealVar* ParFailSignalMassShift = new RooRealVar(
      "ParFailSignalMassShift", "ParFailSignalMassShift", -10.0, 10.0);  //ParFailSignalMassShift.setConstant(kTRUE);
  RooRealVar* ParPassSignalResolution = new RooRealVar(
      "ParPassSignalResolution", "ParPassSignalResolution", 0.0, 10.0);  //ParPassSignalResolution.setConstant(kTRUE);
  RooRealVar* ParFailSignalResolution = new RooRealVar(
      "ParFailSignalResolution", "ParFailSignalResolution", 0.0, 10.0);  //ParFailSignalResolution.setConstant(kTRUE);

  //*********************************************************************************************
  //
  //Load Signal PDFs
  //
  //*********************************************************************************************
  //Load histogram templates
  TFile* Zeelineshape_file = new TFile("MitFitter/templates/60To120Range/ZMassLineshape.root", "READ");
  TH1* histTemplatePass = (TH1D*)Zeelineshape_file->Get(PassSignalTemplateHistName.c_str());
  TH1* histTemplateFail = (TH1D*)Zeelineshape_file->Get(FailSignalTemplateHistName.c_str());

  //Introduce mass shift coordinate transformation
  RooFormulaVar PassShiftedMass("PassShiftedMass", "@0-@1", RooArgList(mass, *ParPassSignalMassShift));
  RooFormulaVar FailShiftedMass("FailShiftedMass", "@0-@1", RooArgList(mass, *ParFailSignalMassShift));

  RooDataHist* dataHistPass = new RooDataHist("dataHistPass", "dataHistPass", RooArgSet(mass), histTemplatePass);
  RooDataHist* dataHistFail = new RooDataHist("dataHistFail", "dataHistFail", RooArgSet(mass), histTemplateFail);
  RooHistPdf* signalShapePassPdf = new RooHistPdf("signalShapePassPdf", "signalShapePassPdf", mass, *dataHistPass, 1);
  RooHistPdf* signalShapeFailPdf = new RooHistPdf("signalShapeFailPdf", "signalShapeFailPdf", mass, *dataHistFail, 1);

  RooFormulaVar* NumSignalPass =
      new RooFormulaVar("NumSignalPass", "ParEfficiency*ParNumSignal", RooArgList(*ParEfficiency, *ParNumSignal));
  RooFormulaVar* NumSignalFail =
      new RooFormulaVar("NumSignalFail", "(1.0-ParEfficiency)*ParNumSignal", RooArgList(*ParEfficiency, *ParNumSignal));

  //*********************************************************************************************
  //
  // Create Background PDFs
  //
  //*********************************************************************************************
  RooExponential* bkgPassPdf = new RooExponential("bkgPassPdf", "bkgPassPdf", mass, *ParPassBackgroundExpCoefficient);
  RooExponential* bkgFailPdf = new RooExponential("bkgFailPdf", "bkgFailPdf", mass, *ParFailBackgroundExpCoefficient);

  //*********************************************************************************************
  //
  // Create Total PDFs
  //
  //*********************************************************************************************
  RooAddPdf pdfPass(
      "pdfPass", "pdfPass", RooArgList(*signalShapePassPdf, *bkgPassPdf), RooArgList(*NumSignalPass, *ParNumBkgPass));
  RooAddPdf pdfFail(
      "pdfFail", "pdfFail", RooArgList(*signalShapeFailPdf, *bkgFailPdf), RooArgList(*NumSignalFail, *ParNumBkgFail));

  // PDF for simultaneous fit
  RooSimultaneous pdfTotal("pdfTotal", "pdfTotal", sample);
  pdfTotal.addPdf(pdfPass, "pdfPass");
  pdfTotal.addPdf(pdfFail, "pdfFail");
  pdfTotal.Print();

  //
  // Define likelihood, add constraints, and run the fit
  //
  RooFitResult* fitResult =
      pdfTotal.fitTo(*dataCombined, RooFit::Save(true), RooFit::Extended(true), RooFit::PrintLevel(-1));
  fitResult->Print("v");

  double nSignalPass = NumSignalPass->getVal();
  double nSignalFail = NumSignalFail->getVal();

  printf("\nFit results:\n");
  printf("    Efficiency = %.4f +- %.4f\n", ParEfficiency->getVal(), ParEfficiency->getPropagatedError(*fitResult));
  cout << "Signal Pass: " << nSignalPass << endl;
  cout << "Signal Fail: " << nSignalFail << endl;

  //--------------------------------------------------------------------------------------------------------------
  // Make plots
  //==============================================================================================================
  RooAbsData::ErrorType errorType = RooAbsData::Poisson;

  TString cname = TString("fit_Pass");
  TCanvas* c = new TCanvas(cname, cname, 800, 600);
  RooPlot* frame1 = mass.frame();
  frame1->SetMinimum(0);
  dataPass->plotOn(frame1, RooFit::DataError(errorType));
  pdfPass.plotOn(frame1, RooFit::ProjWData(*dataPass), RooFit::Components(*bkgPassPdf), RooFit::LineColor(kRed));
  pdfPass.plotOn(frame1, RooFit::ProjWData(*dataPass));
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
  double nsig = ParNumSignal->getVal();
  double nErr = ParNumSignal->getError();
  double e = ParEfficiency->getVal();
  double eErr = ParEfficiency->getError();
  double corr = fitResult->correlation(*ParEfficiency, *ParNumSignal);
  double err = ErrorInProduct(nsig, nErr, e, eErr, corr);
  sprintf(temp, "Signal = %.2f #pm %.2f", NumSignalPass->getVal(), err);
  plotlabel4->AddText(temp);
  TPaveText* plotlabel5 = new TPaveText(0.6, 0.77, 0.8, 0.82, "NDC");
  plotlabel5->SetTextColor(kBlack);
  plotlabel5->SetFillColor(kWhite);
  plotlabel5->SetBorderSize(0);
  plotlabel5->SetTextAlign(12);
  plotlabel5->SetTextSize(0.03);
  sprintf(temp, "Bkg = %.2f #pm %.2f", ParNumBkgPass->getVal(), ParNumBkgPass->getError());
  plotlabel5->AddText(temp);
  TPaveText* plotlabel6 = new TPaveText(0.6, 0.87, 0.8, 0.92, "NDC");
  plotlabel6->SetTextColor(kBlack);
  plotlabel6->SetFillColor(kWhite);
  plotlabel6->SetBorderSize(0);
  plotlabel6->SetTextAlign(12);
  plotlabel6->SetTextSize(0.03);
  plotlabel6->AddText("Passing probes");
  TPaveText* plotlabel7 = new TPaveText(0.6, 0.72, 0.8, 0.77, "NDC");
  plotlabel7->SetTextColor(kBlack);
  plotlabel7->SetFillColor(kWhite);
  plotlabel7->SetBorderSize(0);
  plotlabel7->SetTextAlign(12);
  plotlabel7->SetTextSize(0.03);
  sprintf(temp, "Eff = %.3f #pm %.3f", ParEfficiency->getVal(), ParEfficiency->getErrorHi());
  plotlabel7->AddText(temp);

  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();

  //   c->SaveAs( cname + TString(".eps"));
  c->SaveAs(cname + TString(".gif"));
  delete c;

  cname = TString("fit_Fail");
  TCanvas* c2 = new TCanvas(cname, cname, 500, 500);
  RooPlot* frame2 = mass.frame();
  frame2->SetMinimum(0);
  dataFail->plotOn(frame2, RooFit::DataError(errorType));
  pdfFail.plotOn(frame2, RooFit::ProjWData(*dataFail), RooFit::Components(*bkgFailPdf), RooFit::LineColor(kRed));
  pdfFail.plotOn(frame2, RooFit::ProjWData(*dataFail));
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
  err = ErrorInProduct(nsig, nErr, 1.0 - e, eErr, corr);
  sprintf(temp, "Signal = %.2f #pm %.2f", NumSignalFail->getVal(), err);
  plotlabel4->AddText(temp);
  plotlabel5 = new TPaveText(0.6, 0.77, 0.8, 0.82, "NDC");
  plotlabel5->SetTextColor(kBlack);
  plotlabel5->SetFillColor(kWhite);
  plotlabel5->SetBorderSize(0);
  plotlabel5->SetTextAlign(12);
  plotlabel5->SetTextSize(0.03);
  sprintf(temp, "Bkg = %.2f #pm %.2f", ParNumBkgFail->getVal(), ParNumBkgFail->getError());
  plotlabel5->AddText(temp);
  plotlabel6 = new TPaveText(0.6, 0.87, 0.8, 0.92, "NDC");
  plotlabel6->SetTextColor(kBlack);
  plotlabel6->SetFillColor(kWhite);
  plotlabel6->SetBorderSize(0);
  plotlabel6->SetTextAlign(12);
  plotlabel6->SetTextSize(0.03);
  plotlabel6->AddText("Failing probes");
  plotlabel7 = new TPaveText(0.6, 0.72, 0.8, 0.77, "NDC");
  plotlabel7->SetTextColor(kBlack);
  plotlabel7->SetFillColor(kWhite);
  plotlabel7->SetBorderSize(0);
  plotlabel7->SetTextAlign(12);
  plotlabel7->SetTextSize(0.03);
  sprintf(
      temp, "Eff = %.3f #pm %.3f", ParEfficiency->getVal(), ParEfficiency->getErrorHi(), ParEfficiency->getErrorLo());
  plotlabel7->AddText(temp);

  //   plotlabel->Draw();
  //   plotlabel2->Draw();
  //   plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();

  c2->SaveAs(cname + TString(".eps"));
  c2->SaveAs(cname + TString(".gif"));
  c2->SaveAs(cname + TString(".root"));
  delete c2;

  gBenchmark->Show("fitZCat");
}

void fitZCat() {
  performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/",
             "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36",
             "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36",
             "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf",
             "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf");
}
