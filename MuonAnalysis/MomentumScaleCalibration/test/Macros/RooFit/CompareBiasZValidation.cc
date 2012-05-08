#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TString.h"
#include "TROOT.h"
#include "TLegend.h"

#include "FitMassSlices.cc"
#include "Legend.h"

//gSystem->Load("libRooFitCore");
//gSystem->Load("libRooFit");

class CompareBiasZValidation
{
public:
  CompareBiasZValidation(const TString& leg) 
  {


    gROOT->SetStyle("Plain");

    doFit_ = false;


    TString inputFileName("0_zmumuHisto.root");
    TString outputFileName("BiasCheck.root");


    FitMassSlices fitter;

    fitter.rebinZ = 1; // 2 for Z
    fitter.useChi2 = false;
    // for further rebinning for phi use rebinXphi in FitMassSlices.cc (L20)
    fitter.rebinX = 4;
    fitter.sigma2 = 1.;
    fitter.fit(inputFileName, outputFileName, "voigtian", "", 91, 80, 100, 2, 0.1, 10);

    /*
    file_ = new TFile(outputFile, "READ");

    

    TFile * outputFile = new TFile("CompareBias.root", "RECREATE");
    outputFile->cd();

    compare("MassVsPt", "uniform", 1., 8., "muon pt (GeV)", "Mass (GeV)",leg);
    compare("MassVsEta", "uniform", -2.8, 2.8, "muon #eta", "Mass (GeV)",leg);
    
    compare("MassVsEtaPlus", "uniform", -2.8, 2.8, "muon + #eta", "Mass (GeV)",leg);
    compare("MassVsEtaMinus", "uniform", -2.8, 2.8, "muon - #eta", "Mass (GeV)",leg);   
    compare("MassVsPhiPlus", "sinusoidal", -3.14, 3.14, "muon(+) #phi", "Mass (GeV)",leg);
    compare("MassVsPhiMinus", "sinusoidal", -3.14, 3.14, "muon(-) #phi", "Mass (GeV)",leg);
    
    compare("MassVsEtaPlusMinusDiff", "uniform", -3.3, 3.3, "(#eta neg. muon - #eta pos. muon)", "Mean mass (GeV)",leg); 
    compare("MassVsCosThetaCS", "uniform", -1.1, 1.1, " cos#theta (CS)  ", "Mass (GeV)",leg); 
    compare("MassVsPhiCS", "uniform", -3.14, 3.14, " #phi (CS)  ", "Mass (GeV)",leg); 

    compare("MassVsEtaPhiPlus" , -3.14, 3.14, "positive muon #phi", -2.5, 2.5, "positive muon #eta", "#Delta Mass (GeV)",leg);  
    compare("MassVsEtaPhiMinus", -3.14, 3.14, "negative muon #phi", -2.5, 2.5, "negative muon #eta", "#Delta Mass (GeV)",leg);
    
    outputFile->Write();
    outputFile->Close();
    */
  }
protected:  
  void compare(const TString & histoName, const TString & fitType, const double & xMin, const double & xMax,
	       const TString & xAxisTitle, const TString & yAxisTitle, const TString& leg)
  {
    gDirectory->mkdir(histoName);
    gDirectory->cd(histoName);

    TH1 * histo = (TH1*)getHisto(file_, histoName);
    histo->GetXaxis()->SetTitle(xAxisTitle);
    histo->GetYaxis()->SetTitle(yAxisTitle);
    histo->GetYaxis()->SetTitleOffset(1.25);

    // Fit using RooFit
    // The polynomial in RooFit is a pdf, so it is normalized to unity. This seems to give problems.
    // fitWithRooFit(histo, histoName, fitType, xMin, xMax);
    // Fit with standard root, but then we also need to build the legends.
    if( doFit_ ) {
      fitWithRoot(histo, xMin, xMax, fitType);
    }
    else {
      TCanvas * canvas = drawCanvas(histo, leg, true);
      canvas->Write();
    }
    gDirectory->GetMotherDir()->cd();
  }

  void compare(const TString & histoName, 
		     const double & xMin, const double & xMax, const TString & xAxisTitle, 
		     const double & yMin, const double & yMax, const TString & yAxisTitle, 
		     const TString & zAxisTitle, const TString& leg)
  {
    gDirectory->mkdir(histoName);
    gDirectory->cd(histoName);

    TH2 * histo = (TH2*)getHisto(file_, histoName);
    histo->GetXaxis()->SetTitle(xAxisTitle);
    histo->GetYaxis()->SetTitle(yAxisTitle);
//    histo->GetYaxis()->SetTitleOffset(1.25);

    // Fit using RooFit
    // The polynomial in RooFit is a pdf, so it is normalized to unity. This seems to give problems.
    // fitWithRooFit(histo, histo2, histoName, fitType, xMin, xMax);
    // Fit with standard root, but then we also need to build the legends.
  
    TCanvas * canvas = drawCanvas(histo);
    canvas->Write();
  
    gDirectory->GetMotherDir()->cd();
  }


  TH1* getHisto(TFile * file, const TString & histoName)
  {
    TDirectory* dir = (TDirectory*)file->Get(histoName);
    TCanvas * canvas = (TCanvas*)dir->Get("meanCanvas");
    return (TH1*)canvas->GetPrimitive("meanHisto");
  }

  void fitWithRoot(TH1 * histo, const double & xMin, const double & xMax, const TString & fitType)
  {
    TF1 * f1 = 0;
    if( fitType == "uniform" ) {
      f1 = new TF1("uniform1", "pol0", xMin, xMax);
    }
    else if( fitType == "sinusoidal" ) {
      f1 = new TF1("sinusoidal1", "[0] + [1]*sin([2]*x + [3])", xMin, xMax);
      f1->SetParameter(1, 2.);
      f1->SetParameter(2, 1.);
      f1->SetParameter(3, 1.);
    }
    else {
      std::cout << "Wrong fit type: " << fitType << std::endl;
      exit(1);
    }

    histo->Fit(f1, "", "", xMin, xMax);

    TCanvas * canvas = drawCanvas(histo);

    f1->Draw("same");
    //    TLegend legend;
    //    legend.setText(f1);
    //    legend.Draw("same");

    canvas->Write();
  }

  TCanvas * drawCanvas(TH1 * histo,TString legText="geo 1", const bool addLegend = false)
  {
    TCanvas * canvas = new TCanvas(TString(histo->GetName())+"_canvas", TString(histo->GetName())+" canvas", 1000, 800);
    canvas->Draw();
    canvas->cd();
    histo->Draw();
    histo->SetMarkerStyle(24);
    histo->SetMarkerSize(0.5);


    if( addLegend ) {
      TLegend * leg = new TLegend(0.1,0.7,0.48,0.9);
      leg->AddEntry(histo,legText,"pl");
      leg->Draw("same");
    }

    return canvas;
  }

  void fitWithRooFit(TH1 * histo, const TString & histoName,
		     const TString & fitType, const double & xMin, const double & xMax)
  {
    FitWithRooFit fitter;
    fitter.initConstant(3.097, 3.05, 3.15);
    // fitter.initLinearTerm(0., -1., 1.);

    RooPlot * rooPlot1 = fit( histo, file_->GetName(), &fitter, fitType, xMin, xMax );

    RooRealVar * constant = fitter.constant();
    std::cout << "fitted value for constant 1 = " << constant->getVal() << std::endl;

    TCanvas * canvas = new TCanvas(histoName+"_canvas", histoName+" canvas", 1000, 800);
    canvas->Draw();
    canvas->cd();
    rooPlot1->Draw();
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

  TFile * file_;
  bool doFit_;
};
