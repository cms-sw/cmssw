/**
 * This macro is used to compare the background function from MuScleFit to the effective background distribution seen in input.
 * It needs the result of MuScleFit on three different runs:
 * - all events together
 * - background only
 * - resonance only
 * The first run should be on all events together. Once the parameters for the scale fit are found, they can be introduced as
 * bias for the following two separate runs so as to get the same corrected muons.
 * The following two runs should be made without any fit, they only need to produce the reconstructed mass distributions.
 *
 * This macro takes the reconstructed mass distributions from the last two runs and shows them on the same plot as the all-events
 * distributions. On the same canvas it also shows the background function with the fitted parameters.
 */

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TLegend.h"
#include <vector>
#include "TString.h"

using namespace std;

TH1F * buildHistogram(const double * ResMass, const double * ResHalfWidth, const int xBins, const double & deltaX, const double & xMin, const double & xMax,
                      const int ires, const double & Bgrp1, const double & a, const double & leftWindowFactor, const double & rightWindowFactor, const TH1F* allHisto);

void BackgroundCheck()
{
  TFile * allFile = new TFile("0_MuScleFit.root", "READ");
  TFile * resonanceFile = new TFile("0_MuScleFit.root", "READ");
  TFile * backgroundFile = new TFile("0_MuScleFit.root", "READ");

  TH1F * allHisto = (TH1F*)allFile->Get("hRecBestRes_Mass");
  TH1F * resonanceHisto = (TH1F*)resonanceFile->Get("hRecBestRes_Mass");
  TH1F * backgroundHisto = (TH1F*)backgroundFile->Get("hRecBestRes_Mass");

  int rebinCount = 40;

  allHisto->Rebin(rebinCount);
  resonanceHisto->Rebin(rebinCount);
  backgroundHisto->Rebin(rebinCount);

  double xMin = allHisto->GetXaxis()->GetXmin();
  double xMax = allHisto->GetXaxis()->GetXmax();
  double deltaX = xMax - xMin;

  // The fitted background function gives the background fraction (is normalized).
  // We multiply it by the integral to get the background value.
  int xBins = allHisto->GetNbinsX();

  double ResMass[] = {91.1876, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};
  double ResHalfWidth[] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};
  TString ResName[] = {"Z", "Upsilon3S", "Upsilon2S", "Upsilon1S", "Psi2S", "J/Psi"};

  vector<int> ires;
  vector<double> Bgrp1;
  vector<double> a;
  vector<double> leftWindowFactor;
  vector<double> rightWindowFactor;

  // IMPORTANT: parameters to change
  // -------------------------------
  ires.push_back(3);
  Bgrp1.push_back(0.386119);
  a.push_back(0.111908);
  leftWindowFactor.push_back(10.);
  rightWindowFactor.push_back(10.);

  ires.push_back(5);
  Bgrp1.push_back(0.856432);
  a.push_back(0.407596);
  leftWindowFactor.push_back(10.);
  rightWindowFactor.push_back(10.);

  // -------------------------------

  // Create histograms for the background functions
  vector<TH1F*> backgroundFunctionHisto;
  for( unsigned int i=0; i<ires.size(); ++i ) {
    backgroundFunctionHisto.push_back( buildHistogram(ResMass, ResHalfWidth, xBins, deltaX, xMin, xMax,
                                                      ires[i], Bgrp1[i], a[i], leftWindowFactor[i], rightWindowFactor[i], allHisto) );
  }

  TLegend * legend = new TLegend( 0.55, 0.65, 0.76, 0.82 );
  TCanvas * canvas = new TCanvas("ResMassCanvas", "ResMassCanvas", 1000, 800);
  canvas->cd();
  legend->AddEntry(allHisto, "All events");
  allHisto->SetLineWidth(2);
  allHisto->Draw();

  resonanceHisto->SetLineColor(kRed);
  resonanceHisto->SetLineWidth(2);
  resonanceHisto->Draw("same");
  legend->AddEntry(resonanceHisto, "Resonance events");

  backgroundHisto->SetLineWidth(2);
  backgroundHisto->SetLineColor(kBlue);
  backgroundHisto->Draw("same");
  legend->AddEntry(backgroundHisto, "Background events");

  for( unsigned int i=0; i<ires.size(); ++i ) {
    backgroundFunctionHisto[i]->SetLineWidth(2);
    backgroundFunctionHisto[i]->SetLineColor(kGreen);
    backgroundFunctionHisto[i]->Draw("same");
    legend->AddEntry(backgroundFunctionHisto[i], "Background function for "+ResName[ires[i]]);
  }

  legend->Draw("same");

  canvas->GetPad(0)->SetLogy(true);

  canvas->Print("BackgroundCheck.pdf");
}

TH1F * buildHistogram(const double * ResMass, const double * ResHalfWidth, const int xBins, const double & deltaX, const double & xMin, const double & xMax,
                      const int ires, const double & Bgrp1, const double & a, const double & leftWindowFactor, const double & rightWindowFactor, const TH1F* allHisto)
{
  // For J/Psi exclude the Upsilon from the background normalization as the bin is not used by the fit.
  double lowWindowValue = ResMass[ires]-leftWindowFactor*ResHalfWidth[ires];
  double upWindowValue = ResMass[ires]+rightWindowFactor*ResHalfWidth[ires];

  int lowBin = int((lowWindowValue)*xBins/deltaX);
  int upBin = int((upWindowValue)*xBins/deltaX);

  cout << "lowBin = " << lowBin << ", upBin = " << upBin << endl;
  cout << "lowWindowValue = " << lowWindowValue << ", upWindowValue = " << upWindowValue << endl;

  double xWidth = deltaX/xBins;

  // Exponential
  // -----------
  TF1 * backgroundFunction = new TF1("backgroundFunction", "[0]*([1]*exp(-[1]*x))", xMin, xMax );

  backgroundFunction->SetParameter(0, 1);
  backgroundFunction->SetParameter(1, a);

  TH1F * backgroundFunctionHisto = new TH1F("backgroundFunctionHisto", "backgroundFunctionHisto", xBins, xMin, xMax);
  for( int xBin = 0; xBin < xBins; ++xBin ) {
    // Compute the value in the mean bin point.
    // backgroundFunctionHisto->SetBinContent(xBin+1, backgroundFunction->Eval((xBin+1/2)*xWidth));
    backgroundFunctionHisto->SetBinContent(xBin+1, backgroundFunction->Integral(xBin*xWidth, (xBin+1)*xWidth));
    // cout << "xBin = " << xBin << ", backgroundFunction->Eval((xBin+1/2)*xWidth) = " << backgroundFunction->Eval((xBin+1/2)*xWidth) << endl;
  }

  // Compute the integral used to rescale the background function only in the region actually used for the computation.
  // (Where the function was also normalized, which is also the region the values refer to).
  // double integral = allHisto->Integral(0, lowBin) + allHisto->Integral(upBin, xBins);
  double integral = allHisto->Integral(lowBin, upBin);
  double functionIntegral = backgroundFunction->Integral(lowWindowValue, upWindowValue);
  double functionHistoIntegral = backgroundFunctionHisto->Integral(lowBin, upBin);
  double normalization = integral/functionIntegral*Bgrp1/(upBin-lowBin);
  double normalizationHisto = integral*Bgrp1/functionHistoIntegral;

  // To normalize the function so that its integral in the resonance mass
  // window gives the fraction of events determined by the fit.
  // This is divided by the number of bins in that interval (after rebinning).
  backgroundFunction->SetParameter(0, normalization);

  backgroundFunctionHisto->Scale(normalizationHisto);

  cout << "Integral = " << integral << endl;
  cout << "FunctionHisto integral = " << backgroundFunctionHisto->Integral(lowBin, upBin) << endl;
  cout << "Bgrp1 from histo = " << backgroundFunctionHisto->Integral(lowBin, upBin)/integral << endl;
  cout << "Function integral = " << backgroundFunction->Integral(lowWindowValue, upWindowValue) << endl;
  cout << "Bgrp1 from function = " << backgroundFunction->Integral(lowWindowValue, upWindowValue)/integral << endl;

  return backgroundFunctionHisto;
}
