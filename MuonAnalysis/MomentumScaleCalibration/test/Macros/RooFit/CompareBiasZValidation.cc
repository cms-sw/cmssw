#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TString.h"
#include "TROOT.h"
#include "TLegend.h"

#include "FitMassSlices.cc"
#include "FitMass1D.cc"
#include "Legend.h"

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

    fitter.rebinX = 2; // for further rebinning for phi use rebinXphi in FitMassSlices.cc (L20)
    fitter.rebinY = 2; // default 2
    fitter.rebinZ = 1; // default 2

    fitter.useChi2 = false;
    fitter.sigma2 = 1.;

    double Mmin(75), Mmax(105);
    fitter.fit(inputFileName, outputFileName, "breitWignerTimesCB", "exponentialpol", 91, Mmin, Mmax, 2, 0.1, 10);

    FitMass1D fitMass1D;
    //fitMass1D.rebinX = 10;
    fitMass1D.fitter()->initMean(91, Mmin, Mmax);
    fitMass1D.fitter()->initGamma( 2.4952, 0., 10.);
    fitMass1D.fitter()->gamma()->setConstant(kTRUE);
    fitMass1D.fitter()->initMean2(0., -20., 20.);
    fitMass1D.fitter()->mean2()->setConstant(kTRUE);
    fitMass1D.fitter()->initSigma(1.2, 0., 5.);
    fitMass1D.fitter()->initAlpha(1.5, 0.05, 10.);
    fitMass1D.fitter()->initN(1, 0.01, 100.);
    fitMass1D.fitter()->initExpCoeffA0(-1.,-10.,10.);
    fitMass1D.fitter()->initExpCoeffA1( 0.,-10.,10.);
    fitMass1D.fitter()->initExpCoeffA2( 0., -2., 2.);
    fitMass1D.fitter()->initFsig(0.9, 0., 1.);
    fitMass1D.fitter()->initA0(0., -10., 10.);
    fitMass1D.fitter()->initA1(0., -10., 10.);
    fitMass1D.fitter()->initA2(0., -10., 10.);
    fitMass1D.fitter()->initA3(0., -10., 10.);
    fitMass1D.fitter()->initA4(0., -10., 10.);
    fitMass1D.fitter()->initA5(0., -10., 10.);
    fitMass1D.fitter()->initA6(0., -10., 10.);

    /// Let's fit
    fitMass1D.fit(inputFileName, outputFileName, "UPDATE", Mmin, Mmax, "breitWignerTimesCB", "exponentialpol");

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
    fitter.initA0(3.097, 3.05, 3.15);
    // fitter.initLinearTerm(0., -1., 1.);

    RooPlot * rooPlot1 = fit( histo, file_->GetName(), &fitter, fitType, xMin, xMax );

    RooRealVar * constant = fitter.a0();
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
