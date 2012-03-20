#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TString.h"
#include "TROOT.h"
#include "TLegend.h"


#include "FitMassSlices.cc"
#include "Legend.h"

class CompareBias
{
public:
  CompareBias() : file1_(0), file2_(0)
  {
    gROOT->SetStyle("Plain");

    doFit_ = false;

    TString fileNum1("0");
    TString fileNum2("2");

    TString inputFileName("_MuScleFit.root");
    TString outputFileName("BiasCheck_");

    TString inputFile1(fileNum1+inputFileName);
    TString inputFile2(fileNum2+inputFileName);
    TString outputFile1(outputFileName+fileNum1+".root");
    TString outputFile2(outputFileName+fileNum2+".root");

    FitMassSlices fitter1;
    FitMassSlices fitter2;

    fitter1.rebinZ = 1; // 2 for Z

    fitter1.useChi2 = false;

    fitter1.rebinX = 2;
    fitter1.rebinY = 2;
    fitter1.sigma2 = 1.;
    fitter1.fit(fileNum1+"_MuScleFit.root", "BiasCheck_"+fileNum1+".root", "crystalBall", "exponential", 3.095, 2.8, 3.4, 0.04, 0., 0.);
    // fitter1.fit(fileNum1+"_MuScleFit.root", "BiasCheck_"+fileNum1+".root", "voigtian", "", 91, 80, 100, 2, 0.1, 10);

    if( fileNum2 != "" ) {
      fitter2.rebinX = 2;
      fitter2.rebinY = 2;
      fitter2.sigma2 = 1.;
      fitter2.fit(fileNum2+"_MuScleFit.root", "BiasCheck_"+fileNum2+".root", "crystalBall", "exponential", 3.095, 2.8, 3.4, 0.04, 0., 0.);    
      // fitter2.fit(fileNum2+"_MuScleFit.root", "BiasCheck_"+fileNum2+".root", "voigtian", "", 91, 80, 100, 2, 0.1, 10); 
    }

    file1_ = new TFile(outputFile1, "READ");
    if( fileNum2 != "" ) {
      file2_ = new TFile(outputFile2, "READ");
    }

    TFile * outputFile = new TFile("CompareBias.root", "RECREATE");
    outputFile->cd();

    compare("MassVsPt", "uniform", 1., 8., "muon pt (GeV)", "Mass (GeV)");
    compare("MassVsEta", "uniform", -2.8, 2.8, "muon #eta", "Mass (GeV)");
    
    //r.c. ------------------
    compare("MassVsEtaPlus", "uniform", -2.8, 2.8, "muon + #eta", "Mass (GeV)");
    compare("MassVsEtaMinus", "uniform", -2.8, 2.8, "muon - #eta", "Mass (GeV)");   
    //    compare("MassVsEtaPhiPlus", "uniform", -2.8, 2.8, "muon - #eta #phi", "Mass (GeV)");   
    //r.c. -----------------


    compare("MassVsPhiPlus", "uniform", -3.14, 3.14, "muon(+) #phi", "Mass (GeV)");
    compare("MassVsPhiMinus", "uniform", -3.14, 3.14, "muon(-) #phi", "Mass (GeV)");

    // compare("MassVsPhiPlus", "sinusoidal", -3.14, 3.14, "muon(+) #phi", "Mass (GeV)");
    // compare("MassVsPhiMinus", "sinusoidal", -3.14, 3.14, "muon(-) #phi", "Mass (GeV)");

    outputFile->Write();
    outputFile->Close();
  }
protected:  
  void compare(const TString & histoName, const TString & fitType, const double & xMin, const double & xMax,
	       const TString & xAxisTitle, const TString & yAxisTitle)
  {
    gDirectory->mkdir(histoName);
    gDirectory->cd(histoName);

    TH1 * histo1 = getHisto(file1_, histoName);
    histo1->GetXaxis()->SetTitle(xAxisTitle);
    histo1->GetYaxis()->SetTitle(yAxisTitle);
    histo1->GetYaxis()->SetTitleOffset(1.25);
    TH1 * histo2 = 0;
    if( file2_ != 0 ) {
      histo2 = getHisto(file2_, histoName);
      histo2->GetXaxis()->SetTitle(xAxisTitle);
      histo2->GetYaxis()->SetTitle(yAxisTitle);
      histo2->GetYaxis()->SetTitleOffset(1.25);
    }

    // Fit using RooFit
    // The polynomial in RooFit is a pdf, so it is normalized to unity. This seems to give problems.
    // fitWithRooFit(histo1, histo2, histoName, fitType, xMin, xMax);

    // Fit with standard root, but then we also need to build the legends.
    if( doFit_ ) {
      fitWithRoot(histo1, histo2, xMin, xMax, fitType);
    }
    else {
      TCanvas * canvas = drawCanvas(histo1, histo2, true);
      canvas->Write();
    }
    gDirectory->GetMotherDir()->cd();
  }

  TH1 * getHisto(TFile * file, const TString & histoName)
  {
    TDirectory* dir = (TDirectory*)file->Get(histoName);
    TCanvas * canvas = (TCanvas*)dir->Get("meanCanvas");
    return (TH1*)canvas->GetPrimitive("meanHisto");
  }

  void fitWithRoot(TH1 * histo1, TH1 * histo2, const double & xMin, const double & xMax, const TString & fitType)
  {
    TF1 * f1 = 0;
    TF1 * f2 = 0;
    if( fitType == "uniform" ) {
      f1 = new TF1("uniform1", "pol0", xMin, xMax);
      if( file2_ != 0 ) {
	f2 = new TF1("uniform2", "pol0", xMin, xMax);
      }
    }
    else if( fitType == "sinusoidal" ) {
      f1 = new TF1("sinusoidal1", "[0] + [1]*sin([2]*x + [3])", xMin, xMax);
      f1->SetParameter(1, 2.);
      f1->SetParameter(2, 1.);
      f1->SetParameter(3, 1.);
      if( file2_ != 0 ) {
	f2 = new TF1("sinusoidal2", "[0] + [1]*sin([2]*x + [3])", xMin, xMax);
	f2->SetParameter(1, 2.);
	f2->SetParameter(2, 1.);
	f2->SetParameter(3, 1.);
      }
    }
    else {
      std::cout << "Wrong fit type: " << fitType << std::endl;
      exit(1);
    }

    histo1->Fit(f1, "", "", xMin, xMax);
    if( histo2 != 0 ) {
      histo2->Fit(f2, "", "", xMin, xMax);
    }

    TCanvas * canvas = drawCanvas(histo1, histo2);

    f1->Draw("same");
    if( histo2 != 0 ) {
      f2->Draw("same");
      f2->SetLineColor(kRed);
    }
    TwinLegend legends;
    legends.setText(f1, f2);
    legends.Draw("same");

    canvas->Write();
  }

  TCanvas * drawCanvas(TH1 * histo1, TH1 * histo2, const bool addLegend = false)
  {
    TCanvas * canvas = new TCanvas(TString(histo1->GetName())+"_canvas", TString(histo1->GetName())+" canvas", 1000, 800);
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
      if( addLegend ) {
	TLegend * leg = new TLegend(0.1,0.7,0.48,0.9);
	leg->AddEntry(histo1,"Before calibration","pl");
	leg->AddEntry(histo2,"After calibration","pl");
	leg->Draw("same");
      }
    }

    return canvas;
  }

  void fitWithRooFit(TH1 * histo1, TH1 * histo2, const TString & histoName,
		     const TString & fitType, const double & xMin, const double & xMax)
  {
    FitWithRooFit fitter;
    fitter.initConstant(3.097, 3.05, 3.15);
    // fitter.initLinearTerm(0., -1., 1.);

    RooPlot * rooPlot1 = fit( histo1, file1_->GetName(), &fitter, fitType, xMin, xMax );

    RooRealVar * constant = fitter.constant();
    std::cout << "fitted value for constant 1 = " << constant->getVal() << std::endl;

    RooPlot * rooPlot2 = fit( histo2, file2_->GetName(), &fitter, fitType, xMin, xMax );

    constant = fitter.constant();
    std::cout << "fitted value for constant 2 = " << constant->getVal() << std::endl;

    TCanvas * canvas = new TCanvas(histoName+"_canvas", histoName+" canvas", 1000, 800);
    canvas->Draw();
    canvas->cd();
    rooPlot1->Draw();
    rooPlot2->SetLineColor(kRed);
    rooPlot2->SetMarkerColor(kRed);
    rooPlot2->Draw("same");
    canvas->Write();
  }

  RooPlot * fit(TH1 * histo, const TString & fileName, FitWithRooFit * fitter,
		const TString & fitType, const double & xMin, const double & xMax)
  {
    gDirectory->mkdir(fileName);
    gDirectory->cd(fileName);

    // fitter->fit(histo, "", fitType, xMin, xMax, true);
    fitter->fit(histo, "", fitType, xMin, xMax);
    RooPlot * rooPlot = (RooPlot*)gDirectory->Get(TString(histo->GetName())+"_frame");

    gDirectory->GetMotherDir()->cd();

    return rooPlot;
  }

  TFile * file1_;
  TFile * file2_;
  bool doFit_;
};
