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
#include <iomanip>
#include <fstream>

// RooFit headers
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooCategory.h"
#include "RooArgList.h"
#include "RooDataHist.h"
#include "RooFormulaVar.h"
#include "RooHistPdf.h"
#include "RooAddPdf.h"
#include "RooSimultaneous.h"
#include "RooGaussian.h"
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
#define NBINSPASS 60
#define NBINSFAIL 24

ofstream effTextFile("efficiency.txt");

double ErrorInProduct(double x, double errx, double y, double erry, double corr) {
  double xFrErr = errx / x;
  double yFrErr = erry / y;
  return sqrt(pow(xFrErr, 2) + pow(yFrErr, 2) + 2.0 * corr * xFrErr * yFrErr) * x * y;
}

RooRealVar* LoadParameters(string filename, string label, string parname) {
  RooRealVar* newVar = 0;

  // now read in the parameters from the file.  Stored in the file as
  // parameter manager   |   name   | initial val  | min  | max  | step
  fstream parameter_file(filename.c_str(), ios::in);
  if (!parameter_file) {
    cout << "Error:  Couldn't open parameters file " << filename << endl;
    return false;
  }

  string name;
  string category;
  double initial_val, min, max, step;

  char c;
  Bool_t foundPar = kFALSE;
  while (true) {
    // skip white spaces and look for a #
    while (parameter_file.get(c)) {
      if (isspace(c) || c == '\n')
        continue;
      else if (c == '#')
        while (c != '\n')
          parameter_file.get(c);
      else {
        parameter_file.putback(c);
        break;
      }
    }
    if (parameter_file.fail())
      break;

    parameter_file >> category >> name >> initial_val >> min >> max >> step;
    if (parameter_file.fail())
      break;

    if (category == label && parname == name) {
      // create a new fit parameter
      newVar = new RooRealVar(name.c_str(), name.c_str(), initial_val, min, max);
      if (step == 0) {
        newVar->setConstant(kTRUE);
      }

      break;
    }
  }  //end while loop

  if (!newVar) {
    cout << "Could not load parameter " << parname << " from file " << filename << " , category " << label << endl;
    assert(newVar);
  }

  return newVar;
}

void PrintParameter(RooRealVar* var, string label, string name) {
  cout.width(50);
  cout << left << label;
  cout.width(40);
  cout << left << name;
  cout.precision(5);
  cout.width(15);
  cout << left << scientific << var->getVal();
  cout.width(15);
  cout << left << scientific << var->getMin();
  cout.width(15);
  cout << left << scientific << var->getMax();
  cout.width(15);
  if (var->getAttribute("Constant")) {
    cout << left << scientific << 0.0;
  } else {
    cout << left << scientific << 1.0;
  }
  cout << endl;
}

void performFit(string inputDir,
                string inputParameterFile,
                string label,
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
  const Int_t nbins = 24;

  TString effType = inputDir;

  // The fit variable - lepton invariant mass
  RooRealVar* rooMass_ = new RooRealVar("Mass", "m_{ee}", mlow, mhigh, "GeV/c^{2}");
  RooRealVar Mass = *rooMass_;
  Mass.setBins(nbins);

  // Make the category variable that defines the two fits,
  // namely whether the probe passes or fails the eff criteria.
  RooCategory sample("sample", "");
  sample.defineType("Pass", 1);
  sample.defineType("Fail", 2);

  RooDataSet* dataPass = RooDataSet::read((inputDir + PassInputDataFilename).c_str(), RooArgList(Mass));
  RooDataSet* dataFail = RooDataSet::read((inputDir + FailInputDataFilename).c_str(), RooArgList(Mass));

  RooDataSet* dataCombined = new RooDataSet("dataCombined",
                                            "dataCombined",
                                            RooArgList(Mass),
                                            RooFit::Index(sample),
                                            RooFit::Import("Pass", *dataPass),
                                            RooFit::Import("Fail", *dataFail));

  //*********************************************************************************************
  //Define Free Parameters
  //*********************************************************************************************
  RooRealVar* ParNumSignal = LoadParameters(inputParameterFile, label, "ParNumSignal");
  RooRealVar* ParNumBkgPass = LoadParameters(inputParameterFile, label, "ParNumBkgPass");
  RooRealVar* ParNumBkgFail = LoadParameters(inputParameterFile, label, "ParNumBkgFail");
  RooRealVar* ParEfficiency = LoadParameters(inputParameterFile, label, "ParEfficiency");
  RooRealVar* ParPassBackgroundExpCoefficient =
      LoadParameters(inputParameterFile, label, "ParPassBackgroundExpCoefficient");
  RooRealVar* ParFailBackgroundExpCoefficient =
      LoadParameters(inputParameterFile, label, "ParFailBackgroundExpCoefficient");
  RooRealVar* ParPassSignalMassShift = LoadParameters(inputParameterFile, label, "ParPassSignalMassShift");
  RooRealVar* ParFailSignalMassShift = LoadParameters(inputParameterFile, label, "ParFailSignalMassShift");
  RooRealVar* ParPassSignalResolution = LoadParameters(inputParameterFile, label, "ParPassSignalResolution");
  RooRealVar* ParFailSignalResolution = LoadParameters(inputParameterFile, label, "ParFailSignalResolution");

  // new RooRealVar  ("ParPassSignalMassShift","ParPassSignalMassShift",-2.6079e-02,-10.0, 10.0);   //ParPassSignalMassShift->setConstant(kTRUE);
  //   RooRealVar* ParFailSignalMassShift = new RooRealVar  ("ParFailSignalMassShift","ParFailSignalMassShift",7.2230e-01,-10.0, 10.0);   //ParFailSignalMassShift->setConstant(kTRUE);
  //   RooRealVar* ParPassSignalResolution = new RooRealVar ("ParPassSignalResolution","ParPassSignalResolution",6.9723e-01,0.0, 10.0);     ParPassSignalResolution->setConstant(kTRUE);
  //   RooRealVar* ParFailSignalResolution = new RooRealVar ("ParFailSignalResolution","ParFailSignalResolution",1.6412e+00,0.0, 10.0);     ParFailSignalResolution->setConstant(kTRUE);

  //*********************************************************************************************
  //
  //Load Signal PDFs
  //
  //*********************************************************************************************

  TFile* Zeelineshape_file = new TFile("res/photonEfffromZee.dflag1.eT1.2.gT40.mt15.root", "READ");
  TH1* histTemplatePass = (TH1D*)Zeelineshape_file->Get(PassSignalTemplateHistName.c_str());
  TH1* histTemplateFail = (TH1D*)Zeelineshape_file->Get(FailSignalTemplateHistName.c_str());

  //Introduce mass shift coordinate transformation
  //   RooFormulaVar PassShiftedMass("PassShiftedMass","@0-@1",RooArgList(Mass,*ParPassSignalMassShift));
  //   RooFormulaVar FailShiftedMass("FailShiftedMass","@0-@1",RooArgList(Mass,*ParFailSignalMassShift));

  RooGaussian* PassSignalResolutionFunction = new RooGaussian("PassSignalResolutionFunction",
                                                              "PassSignalResolutionFunction",
                                                              Mass,
                                                              *ParPassSignalMassShift,
                                                              *ParPassSignalResolution);
  RooGaussian* FailSignalResolutionFunction = new RooGaussian("FailSignalResolutionFunction",
                                                              "FailSignalResolutionFunction",
                                                              Mass,
                                                              *ParFailSignalMassShift,
                                                              *ParFailSignalResolution);

  RooDataHist* dataHistPass = new RooDataHist("dataHistPass", "dataHistPass", RooArgSet(Mass), histTemplatePass);
  RooDataHist* dataHistFail = new RooDataHist("dataHistFail", "dataHistFail", RooArgSet(Mass), histTemplateFail);
  RooHistPdf* signalShapePassTemplatePdf =
      new RooHistPdf("signalShapePassTemplatePdf", "signalShapePassTemplatePdf", Mass, *dataHistPass, 1);
  RooHistPdf* signalShapeFailTemplatePdf =
      new RooHistPdf("signalShapeFailTemplatePdf", "signalShapeFailTemplatePdf", Mass, *dataHistFail, 1);

  RooFFTConvPdf* signalShapePassPdf = new RooFFTConvPdf(
      "signalShapePassPdf", "signalShapePassPdf", Mass, *signalShapePassTemplatePdf, *PassSignalResolutionFunction, 2);
  RooFFTConvPdf* signalShapeFailPdf = new RooFFTConvPdf(
      "signalShapeFailPdf", "signalShapeFailPdf", Mass, *signalShapeFailTemplatePdf, *FailSignalResolutionFunction, 2);

  // Now define some efficiency/yield variables
  RooFormulaVar* NumSignalPass =
      new RooFormulaVar("NumSignalPass", "ParEfficiency*ParNumSignal", RooArgList(*ParEfficiency, *ParNumSignal));
  RooFormulaVar* NumSignalFail =
      new RooFormulaVar("NumSignalFail", "(1.0-ParEfficiency)*ParNumSignal", RooArgList(*ParEfficiency, *ParNumSignal));

  //*********************************************************************************************
  //
  // Create Background PDFs
  //
  //*********************************************************************************************
  RooExponential* bkgPassPdf = new RooExponential("bkgPassPdf", "bkgPassPdf", Mass, *ParPassBackgroundExpCoefficient);
  RooExponential* bkgFailPdf = new RooExponential("bkgFailPdf", "bkgFailPdf", Mass, *ParFailBackgroundExpCoefficient);

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
  RooSimultaneous totalPdf("totalPdf", "totalPdf", sample);
  totalPdf.addPdf(pdfPass, "Pass");
  //    totalPdf.Print();
  totalPdf.addPdf(pdfFail, "Fail");
  totalPdf.Print();

  //*********************************************************************************************
  //
  // Perform Fit
  //
  //*********************************************************************************************
  RooFitResult* fitResult = 0;

  // ********* Fix with Migrad first ********** //
  fitResult = totalPdf.fitTo(*dataCombined, RooFit::Save(true), RooFit::Extended(true), RooFit::PrintLevel(-1));
  fitResult->Print("v");

  //   // ********* Fit With Minos ********** //
  //    fitResult = totalPdf.fitTo(*dataCombined, RooFit::Save(true),
  //                               RooFit::Extended(true), RooFit::PrintLevel(-1), RooFit::Minos());
  //    fitResult->Print("v");

  //   // ********* Fix Mass Shift and Fit For Resolution ********** //
  //    ParPassSignalMassShift->setConstant(kTRUE);
  //    ParFailSignalMassShift->setConstant(kTRUE);
  //    ParPassSignalResolution->setConstant(kFALSE);
  //    ParFailSignalResolution->setConstant(kFALSE);
  //    fitResult = totalPdf.fitTo(*dataCombined, RooFit::Save(true),
  //    RooFit::Extended(true), RooFit::PrintLevel(-1));
  //    fitResult->Print("v");

  //   // ********* Do Final Fit ********** //
  //    ParPassSignalMassShift->setConstant(kFALSE);
  //    ParFailSignalMassShift->setConstant(kFALSE);
  //    ParPassSignalResolution->setConstant(kTRUE);
  //    ParFailSignalResolution->setConstant(kTRUE);
  //    fitResult = totalPdf.fitTo(*dataCombined, RooFit::Save(true),
  //                                             RooFit::Extended(true), RooFit::PrintLevel(-1));
  //    fitResult->Print("v");

  double nSignalPass = NumSignalPass->getVal();
  double nSignalFail = NumSignalFail->getVal();
  double denominator = nSignalPass + nSignalFail;

  printf("\nFit results:\n");
  if (fitResult->status() != 0) {
    std::cout << "ERROR: BAD FIT STATUS" << std::endl;
  }

  printf("    Efficiency = %.4f +- %.4f\n", ParEfficiency->getVal(), ParEfficiency->getPropagatedError(*fitResult));
  cout << "Signal Pass: " << nSignalPass << endl;
  cout << "Signal Fail: " << nSignalFail << endl;

  cout << "*********************************************************************\n";
  cout << "Final Parameters\n";
  cout << "*********************************************************************\n";
  PrintParameter(ParNumSignal, label, "ParNumSignal");
  PrintParameter(ParNumBkgPass, label, "ParNumBkgPass");
  PrintParameter(ParNumBkgFail, label, "ParNumBkgFail");
  PrintParameter(ParEfficiency, label, "ParEfficiency");
  PrintParameter(ParPassBackgroundExpCoefficient, label, "ParPassBackgroundExpCoefficient");
  PrintParameter(ParFailBackgroundExpCoefficient, label, "ParFailBackgroundExpCoefficient");
  PrintParameter(ParPassSignalMassShift, label, "ParPassSignalMassShift");
  PrintParameter(ParFailSignalMassShift, label, "ParFailSignalMassShift");
  PrintParameter(ParPassSignalResolution, label, "ParPassSignalResolution");
  PrintParameter(ParFailSignalResolution, label, "ParFailSignalResolution");
  cout << endl << endl;

  //--------------------------------------------------------------------------------------------------------------
  // Make plots
  //==============================================================================================================
  TFile* canvasFile = new TFile("Efficiency_FitResults.root", "UPDATE");

  RooAbsData::ErrorType errorType = RooAbsData::Poisson;

  Mass.setBins(NBINSPASS);
  TString cname = TString((label + "_Pass").c_str());
  TCanvas* c = new TCanvas(cname, cname, 800, 600);
  RooPlot* frame1 = Mass.frame();
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
  TPaveText* plotlabel8 = new TPaveText(0.6, 0.72, 0.8, 0.66, "NDC");
  plotlabel8->SetTextColor(kBlack);
  plotlabel8->SetFillColor(kWhite);
  plotlabel8->SetBorderSize(0);
  plotlabel8->SetTextAlign(12);
  plotlabel8->SetTextSize(0.03);
  sprintf(temp, "#chi^{2}/DOF = %.3f", frame1->chiSquare());
  plotlabel8->AddText(temp);

  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();
  plotlabel8->Draw();

  //   c->SaveAs( cname + TString(".eps"));
  c->SaveAs(cname + TString(".gif"));
  canvasFile->WriteTObject(c, c->GetName(), "WriteDelete");

  Mass.setBins(NBINSFAIL);
  cname = TString((label + "_Fail").c_str());
  TCanvas* c2 = new TCanvas(cname, cname, 800, 600);
  RooPlot* frame2 = Mass.frame();
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
  plotlabel8 = new TPaveText(0.6, 0.72, 0.8, 0.66, "NDC");
  plotlabel8->SetTextColor(kBlack);
  plotlabel8->SetFillColor(kWhite);
  plotlabel8->SetBorderSize(0);
  plotlabel8->SetTextAlign(12);
  plotlabel8->SetTextSize(0.03);
  sprintf(temp, "#chi^{2}/DOF = %.3f", frame2->chiSquare());
  plotlabel8->AddText(temp);

  //   plotlabel->Draw();
  //   plotlabel2->Draw();
  //   plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();
  plotlabel8->Draw();

  c2->SaveAs(cname + TString(".gif"));
  //   c2->SaveAs( cname + TString(".eps"));
  //   c2->SaveAs( cname + TString(".root"));
  canvasFile->WriteTObject(c2, c2->GetName(), "WriteDelete");

  canvasFile->Close();

  effTextFile.width(40);
  effTextFile << label;
  effTextFile.width(20);
  effTextFile << setiosflags(ios::fixed) << setprecision(4) << left << ParEfficiency->getVal();
  effTextFile.width(20);
  effTextFile << left << ParEfficiency->getErrorHi();
  effTextFile.width(20);
  effTextFile << left << ParEfficiency->getErrorLo();
  effTextFile.width(14);
  effTextFile << setiosflags(ios::fixed) << setprecision(2) << left << nSignalPass;
  effTextFile.width(14);
  effTextFile << left << nSignalFail << endl;
}

void ElectronTagAndProbeFitter() {
  // //////////////////////////////////////////////////////////
  effTextFile.width(40);
  effTextFile << left << "Type";
  effTextFile.width(20);
  effTextFile << left << "Efficiency";
  effTextFile.width(20);
  effTextFile << left << " Uncertainty(+) ";
  effTextFile.width(20);
  effTextFile << left << " Uncertainty(-) ";
  effTextFile.width(14);
  effTextFile << left << "NPass";
  effTextFile.width(14);
  effTextFile << left << "NFail" << endl;

  // //////////////////////////////////////////////////////////

  // //////////////////////////////////////////////////////////
  //   //  Super cluster --> gsfElectron efficiency
  // //////////////////////////////////////////////////////////

  //**************
  //User Pass Template For Fail Sample - TPTree
  //**************
  //       performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco", "Mass_TagPlusSCPassReco_Data_36", "Mass_TagPlusSCFailReco_Data_36", "Mass_TagPlusSCPassReco", "Mass_TagPlusSCPassReco" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCPassReco_EB_Pt20ToInf" );
  //       performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCPassReco_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB_Minus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCPassReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE_Minus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCPassReco_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB_Plus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCPassReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE_Plus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCPassReco_EE_Pt20ToInf" );

  //**************
  //User Fail Template For Fail Sample - TPTree
  //**************
  //      performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco", "Mass_TagPlusSCPassReco_Data_36", "Mass_TagPlusSCFailReco_Data_36", "Mass_TagPlusSCPassReco", "Mass_TagPlusSCFailReco" );
  //      performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //      performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB_Minus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE_Minus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB_Plus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE_Plus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );

  //**************
  //Impose WP95 Iso Cut on Photon probes - TPTree  SYstematics
  //**************
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80_WithPhotonIsoCut/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco", "Mass_TagPlusSCPassReco_Data_36", "Mass_TagPlusSCFailReco_Data_36", "Mass_TagPlusSCPassReco", "Mass_TagPlusSCFailReco" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80_WithPhotonIsoCut/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80_WithPhotonIsoCut/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );

  //**************
  //Impose WP95 Iso Cut on Photon probes - BAMBU
  //**************
  //      performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco", "Mass_TagPlusSCPassReco_Data_36", "Mass_TagPlusSCFailReco_Data_36", "Mass_TagPlusSCPassReco", "Mass_TagPlusSCFailReco" );
  //      performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //      performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB_Minus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE_Minus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EB_Plus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCFailReco_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCPassReco_EB_Pt20ToInf", "Mass_TagPlusSCFailReco_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_SCToReco_EE_Plus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCFailReco_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusSCPassReco_EE_Pt20ToInf", "Mass_TagPlusSCFailReco_EE_Pt20ToInf" );

  // //////////////////////////////////////////////////////////
  //   //  gsfElectron --> WP-95 selection efficiency
  // //////////////////////////////////////////////////////////

  //**************
  //TAG 80 - TPTree
  //**************
  //    performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95", "Mass_TagPlusRecoPassVBTF95IdIso_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso", "Mass_TagPlusRecoFailVBTF95IdIso" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );

  //**************
  //TAG 80 - BAMBU
  //**************
  //    performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95", "Mass_TagPlusRecoPassVBTF95IdIso_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso", "Mass_TagPlusRecoFailVBTF95IdIso" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );

  // //**************
  // //TAG 95 - BAMBU
  // //**************
  //    performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95", "Mass_TagPlusRecoPassVBTF95IdIso_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso", "Mass_TagPlusRecoFailVBTF95IdIso" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP95/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );

  //**************
  //TAG 60 - BAMBU
  //**************
  //    performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95", "Mass_TagPlusRecoPassVBTF95IdIso_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso", "Mass_TagPlusRecoFailVBTF95IdIso" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP60/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );

  //       performFit("EfficiencyFitter/input/Data_36_09122010_JW_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EB", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EB_Pt20ToInf" );
  //       performFit("EfficiencyFitter/input/Data_36_09122010_JW_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF95_EE", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF95IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF95IdIso_EE_Pt20ToInf" );

  cout << "########################################" << endl;

  // // //////////////////////////////////////////////////////////
  // //   //  gsfElectron --> WP-80 selection efficiency
  // // //////////////////////////////////////////////////////////

  //**************
  //TAG 80 - TP
  //**************
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80", "Mass_TagPlusRecoPassVBTF80IdIso_Data_36", "Mass_TagPlusRecoFailVBTF80IdIso_Data_36", "Mass_TagPlusRecoPassVBTF80IdIso", "Mass_TagPlusRecoFailVBTF80IdIso" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EB", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EE", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EB_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EE_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EB_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TP_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EE_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf" );

  //**************
  //TAG 80 - BAMBU
  //**************

  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80", "Mass_TagPlusRecoPassVBTF80IdIso_Data_36", "Mass_TagPlusRecoFailVBTF80IdIso_Data_36", "Mass_TagPlusRecoPassVBTF80IdIso", "Mass_TagPlusRecoFailVBTF80IdIso" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EB", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EE", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf_Data_36", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EB_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EE_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf_Data_36_Minus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EB_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EB_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EB_Pt20ToInf" );
  //     performFit("EfficiencyFitter/input/Data_36_09122010_TagWP80/", "EfficiencyFitter/results/Parameters.txt", "Efficiency_RecoToVBTF80_EE_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf_Data_36_Plus", "Mass_TagPlusRecoPassVBTF80IdIso_EE_Pt20ToInf", "Mass_TagPlusRecoFailVBTF80IdIso_EE_Pt20ToInf" );

  performFit("res/",
             "results/Parameters.txt",
             "Efficiency_Photon",
             "photonEfffromZee.passbar.dflag1.eT1.2.gT40.mt15.ptbin0.txt",
             "photonEfffromZee.failbar.dflag1.eT1.2.gT40.mt15.ptbin0.txt",
             "hh_Meg_barrel_mc_pt_0",
             "hh_Megf_barrel_mc_pt_0");

  // // //////////////////////////////////////////////////////////
  // //   //   WP-95 --> HLT triggering efficiency
  // // //////////////////////////////////////////////////////////

  //    cout << "########################################" << endl;

  // // //////////////////////////////////////////////////////////
  // //   //   WP-80 --> HLT triggering efficiency
  // // //////////////////////////////////////////////////////////

  //    cout << "########################################" << endl;

  effTextFile.close();
}
