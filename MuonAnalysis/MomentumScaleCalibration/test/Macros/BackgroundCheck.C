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

using namespace std;

void BackgroundCheck()
{

  TFile * allFile = new TFile("0_MuScleFit.root", "READ");
  TFile * resonanceFile = new TFile("resonance/0_MuScleFit.root", "READ");
  TFile * backgroundFile = new TFile("background/0_MuScleFit.root", "READ");

  TH1F * allHisto = (TH1F*)allFile->Get("hRecBestRes_Mass");
  TH1F * resonanceHisto = (TH1F*)resonanceFile->Get("hRecBestRes_Mass");
  TH1F * backgroundHisto = (TH1F*)backgroundFile->Get("hRecBestRes_Mass");

  int rebinCount = 200;

  allHisto->Rebin(rebinCount);
  resonanceHisto->Rebin(rebinCount);
  backgroundHisto->Rebin(rebinCount);

  // The fitted background function gives the background fraction (is normalized).
  // We multiply it by the integral to get the background value.
  int xBins = allHisto->GetNbinsX();

  double ResMass[] = {91.1876, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};

  double ResHalfWidth[] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};


  // IMPORTANT: parameters to change
  // -------------------------------
  int ires = 3;
  double Bgrp1 = 0.0260452;
  double a = 0.105061;
  // -------------------------------

  // For J/Psi exclude the Upsilon from the background normalization as the bin is not used by the fit.
  double lowWindowValue = ResMass[ires]-ResHalfWidth[ires];
  double upWindowValue = ResMass[ires]+ResHalfWidth[ires];

  // ATTENTION: for the Z the number of bins is more than 30.

  int lowBin = int((lowWindowValue)*xBins/30.);
  int upBin = int((upWindowValue)*xBins/30.);

  // Constant
  // --------
  // TF1 * backgroundFunction = new TF1("backgroundFunction","[0]",allHisto->GetXaxis()->GetXmin(),allHisto->GetXaxis()->GetXmax());
  // backgroundFunction->SetParameter(0, 0.216393*integral);

  // Exponential
  // -----------
  TF1 * backgroundFunction = new TF1("backgroundFunction","[0]*([1]*exp(-[1]*x))",allHisto->GetXaxis()->GetXmin(),allHisto->GetXaxis()->GetXmax());

  backgroundFunction->SetParameter(0, 1);
  backgroundFunction->SetParameter(1, a);

  // Compute the integral used to rescale the background function only in the region actually used for the computation.
  // (Where the function was also normalized, which is also the region the values refer to).
  // double integral = allHisto->Integral(0, lowBin) + allHisto->Integral(upBin, xBins);
  double integral = allHisto->Integral(lowBin, upBin);
  double functionIntegral = backgroundFunction->Integral(lowWindowValue, upWindowValue);
  double normalization = integral/functionIntegral*Bgrp1/(upBin-lowBin);

  // To normalize the function so that its integral in the resonance mass
  // window gives the fraction of events determined by the fit.
  // This is divided by the number of bins in that interval (after rebinning).
  backgroundFunction->SetParameter(0, normalization);

  cout << "Integral = " << integral << endl;

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

  backgroundFunction->SetLineColor(kGreen);
  backgroundFunction->Draw("same");
  legend->AddEntry(backgroundFunction, "Background function");

  legend->Draw("same");

  canvas->GetPad(0)->SetLogy(true);

  canvas->Print("BackgroundCheck.pdf");
}
