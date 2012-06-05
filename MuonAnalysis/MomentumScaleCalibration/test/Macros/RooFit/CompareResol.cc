#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TProfile.h"
#include "TString.h"
#include "TROOT.h"

#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitResolSlices.cc"
#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitMassSlices.cc"
#include "TLegend.h"

TH1 * getHisto(TDirectory * dir)
{
  TCanvas * canvas = (TCanvas*)dir->Get("sigmaCanvas");
  return (TH1*)canvas->GetPrimitive("sigmaHisto");
}

void drawCanvas(const TString & canvasName, TH1 * histo1, TH1 * histo2, const TString & histo1LegendTitle, const TString & histo2LegendTitle)
{
  TCanvas * canvas = new TCanvas(canvasName, canvasName, 1000, 800);
  canvas->Draw();
  canvas->cd();
  histo1->Draw();
  histo1->SetMarkerStyle(24);
  histo1->SetMarkerSize(0.5);

  if( histo2 != 0 ) {
    histo2->SetLineColor(kRed);
    histo2->SetMarkerColor(kRed);
    histo2->SetMarkerSize(0.5);
    histo2->SetMarkerStyle(24);
    histo2->Draw("same");

    TLegend * legend = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend->SetFillColor(0);
    legend->SetTextColor(1);
    legend->SetTextSize(0.02);
    legend->SetBorderSize(1);
    legend->AddEntry(histo1, histo1LegendTitle);
    legend->AddEntry(histo2, histo2LegendTitle);
    legend->Draw("same");
  }

  canvas->Write();
}

struct ConfCompare
{
  ConfCompare() :
    file(0),
    takeProfile(false),
    MC(false)
  {}

  TString inputFile;
  TString dirName;
  TString funcHistoName;
  TFile * file;
  TString xAxisTitle;
  TString yAxisTitle;
  TString mainDirName;
  TString subDirName;
  TString funcLegendTitle;
  TString histoLegendTitle;
  bool takeProfile;
  bool MC;

  void compare()
  {
    gDirectory->mkdir(dirName);
    gDirectory->cd(dirName);
    TDirectory * outputDir = gDirectory;

    // Resolutions from fitted function
    TFile * mainFile = new TFile(inputFile, "READ");
    TProfile * funcHisto = 0;
    if( takeProfile ) {
      TH2 * func2Dhisto = (TH2*)mainFile->FindObjectAny(funcHistoName);
      funcHisto = func2Dhisto->ProfileX();
    }
    else {
      funcHisto = (TProfile*)mainFile->FindObjectAny(funcHistoName);
    }
    outputDir->cd();

    TH1 * histo = 0;
    // In this case we take the profile histogram
    if( MC || takeProfile ) {
      TDirectory* dir = (TDirectory*)file->Get(mainDirName);
      TDirectory * subDir = (TDirectory*)dir->Get(subDirName);
      histo = getHisto(subDir);
    }

    funcHisto->GetXaxis()->SetTitle(xAxisTitle);
    funcHisto->GetYaxis()->SetTitle(yAxisTitle);
    funcHisto->GetYaxis()->SetTitleOffset(1.25);

    drawCanvas(inputFile+dirName+funcHistoName+mainDirName+"_canvas", funcHisto, histo, funcLegendTitle, histoLegendTitle);

    gDirectory->GetMotherDir()->cd();
  }
};

class CompareResol
{
public:
  CompareResol() : file1_(0), file2_(0)
  {
    gROOT->SetStyle("Plain");

    MC_ = true;

    TString fileNum1("0");
    TString fileNum2("3");

    TString inputFileName("_MuScleFit.root");
    TString outputFileName("ResolCheck_");

    TString inputFile1(fileNum1+inputFileName);
    TString inputFile2(fileNum2+inputFileName);
    TString outputFileName1(outputFileName+fileNum1+".root");
    TString outputFileName2(outputFileName+fileNum2+".root");

    TFile * outputFile1 = new TFile(outputFileName1, "RECREATE");
    TFile * outputFile2 = new TFile(outputFileName2, "RECREATE");

    // Resolutions from MC comparison
    if( MC_ ) {
      FitResolSlices resolFitter1;
      FitResolSlices resolFitter2;

      // muon Pt resolution vs muon Pt and Eta
      resolFitter1.fit(fileNum1+"_MuScleFit.root", "ResolCheck_"+fileNum1+".root", "gaussian",
		       0., -0.1, 0.1, 0.03, 0., 0.1, "hResolPtGenVSMu_ResoVS", "ResolPtVs", outputFile1);
      if( fileNum2 != "" ) {
	resolFitter2.fit(fileNum2+"_MuScleFit.root", "ResolCheck_"+fileNum2+".root", "gaussian",
			 0., -0.1, 0.1, 0.03, 0., 0.1, "hResolPtGenVSMu_ResoVS", "ResolPtVs", outputFile2);
      }

      // Resonance mass resolution vs muon Pt and Eta
      resolFitter1.fit(fileNum1+"_MuScleFit.root", "ResolCheck_"+fileNum1+".root", "gaussian",
		       0., -0.1, 0.1, 0.03, 0., 0.1, "DeltaMassOverGenMassVs", "ResolMassVsMuon", outputFile1);
      if( fileNum2 != "" ) {
	resolFitter2.fit(fileNum2+"_MuScleFit.root", "ResolCheck_"+fileNum2+".root", "gaussian",
			 0., -0.1, 0.1, 0.03, 0., 0.1, "DeltaMassOverGenMassVs", "ResolMassVsMuon", outputFile2);
      }
    }

    // Resolutions for mass from mass fits
    FitMassSlices massFitter1;
    FitMassSlices massFitter2;

    TDirectory * massDir1 = outputFile1->mkdir("MassResol");
    massFitter1.rebinX = 4;
    massFitter1.fit(fileNum1+"_MuScleFit.root", "ResolCheck_"+fileNum1+".root", "gaussian", "exponential",
    		    3.1, 3., 3.2, 0.03, 0., 0.1, massDir1);
    if( fileNum2 != "" ) {
      TDirectory * massDir2 = outputFile2->mkdir("MassResol");
      massFitter2.rebinX = 4;
      massFitter2.fit(fileNum2+"_MuScleFit.root", "ResolCheck_"+fileNum2+".root", "gaussian", "exponential",
		      3.1, 3., 3.2, 0.03, 0., 0.1, massDir2);
    }

    outputFile1->Write();
    outputFile1->Close();
    outputFile2->Write();
    outputFile2->Close();

    // Reading back the closed files to do comparisons
    file1_ = new TFile(outputFileName1, "READ");
    if( fileNum2 != "" ) {
      file2_ = new TFile(outputFileName2, "READ");
    }

    TFile * outputFile = new TFile("CompareResol.root", "RECREATE");
    outputFile->cd();

    // Mass resolution
    // ---------------
    if( MC_ ) {
      // Comparison of MC resolution before-after the fit
      compareBeforeAfter("DeltaMassOverGenMassVs", "ResolMassVsMuonPt", "muon pt (GeV)", "#sigmaM/M");
      compareBeforeAfter("DeltaMassOverGenMassVs", "ResolMassVsMuonEta", "muon #eta", "#sigmaM/M");
    }
    // Comparison of mass resolution from data before and after the fit
    compareBeforeAfter("MassResol", "MassVsPt", "muon pt (GeV)", "#sigmaM");
    compareBeforeAfter("MassResol", "MassVsEta", "muon #eta", "#sigmaM");

    // Make mass comparisons for both files
    makeAllMassComparisons(inputFile1, fileNum1, file1_);
    if( file2_ != 0 ) {
      makeAllMassComparisons(inputFile2, fileNum2, file2_);
    }

    // Muon resolution
    // ---------------
    if( MC_ ) {
      // Muon Pt resolution vs Pt and Eta compared to MC resolution
      compareBeforeAfter("hResolPtGenVSMu_ResoVS", "ResolPtVsPt", "muon pt (GeV)", "#sigmaM");
      compareBeforeAfter("hResolPtGenVSMu_ResoVS", "ResolPtVsEta", "muon pt (GeV)", "#sigmaM");
    }

    // Make sigmaPt/Pt comparisons for both files
    makeAllPtComparisons(inputFile1, fileNum1, file1_);
    if( file2_ != 0 ) {
      makeAllPtComparisons(inputFile2, fileNum2, file2_);
    }

    outputFile->Write();
    outputFile->Close();
  }
protected:  
  void compareBeforeAfter(const TString & mainDirName, const TString & subDirName,
			  const TString & xAxisTitle, const TString & yAxisTitle)
  {
    gDirectory->mkdir(mainDirName);
    gDirectory->cd(mainDirName);

    TDirectory* dir1 = (TDirectory*)file1_->Get(mainDirName);
    TDirectory * subDir1 = (TDirectory*)dir1->Get(subDirName);
    TH1 * histo1 = getHisto(subDir1);
    histo1->GetXaxis()->SetTitle(xAxisTitle);
    histo1->GetYaxis()->SetTitle(yAxisTitle);
    histo1->GetYaxis()->SetTitleOffset(1.25);

    TH1 * histo2 = 0;
    if( file2_ != 0 ) {
      TDirectory* dir2 = (TDirectory*)file2_->Get(mainDirName);
      TDirectory * subDir2 = (TDirectory*)dir2->Get(subDirName);
      histo2 = getHisto(subDir2);
      histo2->GetXaxis()->SetTitle(xAxisTitle);
      histo2->GetYaxis()->SetTitle(yAxisTitle);
      histo2->GetYaxis()->SetTitleOffset(1.25);
    }

    drawCanvas(mainDirName+subDirName+"_canvas", histo1, histo2, histo1->GetTitle(), histo2->GetTitle());

    gDirectory->GetMotherDir()->cd();
  }

  void makeAllMassComparisons(const TString & inputFile, const TString & fileNum, TFile * mainFile)
  {
    // Mass relative resolution Vs Pt compared to MC resolution

    ConfCompare conf;
    conf.inputFile = inputFile;
    conf.dirName = "FunctionMassRelativeResol_"+fileNum;
    conf.funcHistoName = "hFunctionResolMassVSMu_ResoVSPt_prof";
    conf.file = mainFile;
    conf.xAxisTitle = "muon pt (GeV)";
    conf.yAxisTitle = "#sigmaM/M";
    conf.mainDirName = "DeltaMassOverGenMassVs";
    conf.subDirName = "ResolMassVsMuonPt";
    conf.funcLegendTitle = "Mass relative resolution vs muon Pt from fitted function";
    conf.histoLegendTitle = "Mass relative resolution vs muon Pt from MC";
    conf.MC = MC_;
    conf.compare();

    // Mass relative resolution Vs Eta compared to MC resolution
    conf.funcHistoName = "hFunctionResolMassVSMu_ResoVSEta_prof";
    conf.xAxisTitle = "muon #eta";
    conf.subDirName = "ResolMassVsMuonEta";
    conf.funcLegendTitle = "Mass relative resolution vs muon #eta from fitted function";
    conf.histoLegendTitle = "Mass relative resolution vs muon #eta from MC";
    conf.compare();

    // Mass resolution Vs Pt compared to resolution from sigma of gaussian fits on data
    conf.dirName = "FunctionMassResol_"+fileNum;
    conf.funcHistoName = "hResolMassVSMu_ResoVSPt";
    conf.xAxisTitle = "muon pt (GeV)";
    conf.yAxisTitle = "#sigmaM";
    conf.mainDirName = "MassResol";
    conf.subDirName = "MassVsPt";
    conf.funcLegendTitle = "Mass resolution vs muon Pt from fitted function";
    conf.histoLegendTitle = "Mass relative resolution vs muon Pt from mass fits";
    conf.takeProfile = true;
    conf.compare();

    // Mass resolution Vs Eta compared to resolution from sigma of gaussian fits on data
    conf.funcHistoName = "hResolMassVSMu_ResoVSEta";
    conf.xAxisTitle = "muon #eta";
    conf.yAxisTitle = "#sigmaM";
    conf.subDirName = "MassVsEta";
    conf.funcLegendTitle = "Mass resolution vs muon #eta from fitted function";
    conf.histoLegendTitle = "Mass relative resolution vs muon #eta from mass fits";
    conf.compare();
  }


  void makeAllPtComparisons(const TString & inputFile, const TString & fileNum, TFile * mainFile)
  {
    // Pt relative resolution Vs Pt compared to MC resolution
    ConfCompare conf;
    conf.inputFile = inputFile;
    conf.dirName = "FunctionPtRelativeResol_"+fileNum;
    conf.funcHistoName = "hFunctionResolPt_ResoVSPt_prof";
    conf.file = mainFile;
    conf.xAxisTitle = "muon pt (GeV)";
    conf.yAxisTitle = "#sigmaPt/Pt";
    conf.mainDirName = "hResolPtGenVSMu_ResoVS";
    conf.subDirName = "ResolPtVsPt";
    conf.funcLegendTitle = "Muon Pt relative resolution vs muon Pt from fitted function";
    conf.histoLegendTitle = "Muon Pt relative resolution vs muon Pt from MC";
    conf.MC = MC_;
    conf.compare();

    // Pt relative resolution Vs Eta compared to MC resolution
    conf.dirName = "FunctionEtaRelativeResol_"+fileNum;
    conf.funcHistoName = "hFunctionResolPt_ResoVSEta_prof";
    conf.xAxisTitle = "muon #eta";
    conf.subDirName = "ResolPtVsEta";
    conf.funcLegendTitle = "Muon Pt relative resolution vs muon #eta from fitted function";
    conf.histoLegendTitle = "Muon Pt relative resolution vs muon #eta from MC";
    conf.compare();
  }

  RooPlot * fit(TH1 * histo, const TString & fileName, FitWithRooFit * fitter,
		const TString & fitType, const double & xMin, const double & xMax)
  {
    gDirectory->mkdir(fileName);
    gDirectory->cd(fileName);

    fitter->fit(histo, "", fitType, xMin, xMax);
    RooPlot * rooPlot = (RooPlot*)gDirectory->Get(TString(histo->GetName())+"_frame");

    gDirectory->GetMotherDir()->cd();

    return rooPlot;
  }

  TFile * file1_;
  TFile * file2_;

  bool MC_;
};
