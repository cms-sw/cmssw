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
#include <sstream>


// Linear function as in Functions.h.
// Note that if the type1 background function in Functions.h changes this should be updated too.
class Linear
{
 public:
  Linear(const double & lowerLimit, const double & upperLimit) : lowerLimit_(lowerLimit), upperLimit_(upperLimit) {}

  double operator()( const double * x, const double * parval ) const
  {
    double a = parval[1];
    double b = parval[2];

    double norm = -(a*lowerLimit_ + b*lowerLimit_*lowerLimit_/2.);

    if( -a/b > upperLimit_ ) norm += a*upperLimit_ + b*upperLimit_*upperLimit_/2.;
    else norm += -a*a/(2*b);

    if( x[0] < -a/b && norm != 0 ) return (a + b*x[0])/norm;
    else return 0;
  }
protected:
  double lowerLimit_, upperLimit_;
};

class Atan
{
public:
  Atan(const double & lowerLimit, const double & upperLimit) : lowerLimit_(lowerLimit), upperLimit_(upperLimit) {}

  double operator()( const double * x, const double * parval ) const
  {
    double value = atan(parval[1]*(x[0] + parval[2]));
    double upperX = parval[1]*(upperLimit_ + parval[2]);
    double lowerX = parval[1]*(lowerLimit_ + parval[2]);
    double norm = (upperX*atan(upperX) - 0.5*log(1+upperX*upperX) - (lowerX*atan(lowerX) - 0.5*log(1+lowerX*lowerX)))/parval[1];

    if( norm != 0 ) return value/norm;
    return 0.;
  }
protected:
  double lowerLimit_, upperLimit_;
};

TH1F * subRangeHisto( const double * resMass, const double * resHalfWidth,
                      const int iRes, TH1F * histo, const TString & name )
{
  std::stringstream ss;
  ss << iRes;

  TH1F* newHisto = (TH1F*)histo->Clone(name+ss.str());
  newHisto->SetAxisRange( resMass[iRes] - resHalfWidth[iRes], resMass[iRes] + resHalfWidth[iRes] );
  // Workaround for when we fit all the Upsilon resonances together
  if( iRes == 3 ) {
    newHisto->SetAxisRange( resMass[iRes] - resHalfWidth[iRes], resMass[1] + resHalfWidth[iRes] );
  }
  return newHisto;
}

TH1F * buildHistogram(const double * ResMass, const double * ResHalfWidth, const int xBins, const double & deltaX, const double & xMin, const double & xMax,
                      const int ires, const double & Bgrp1, const double & a, const double & leftWindowBorder, const double & rightWindowBorder,
		      const TH1F* allHisto, const int backgroundType, const double & b = 0);

void BackgroundCheck()
{
  TFile * allFile = new TFile("0_MuScleFit.root", "READ");
  TFile * resonanceFile = new TFile("0_MuScleFit.root", "READ");
  TFile * backgroundFile = new TFile("0_MuScleFit.root", "READ");

  // TH1F * allHisto = (TH1F*)allFile->Get("hRecBestRes_Mass");
  TH1F * allHisto = (TH1F*)allFile->Get("hRecBestResAllEvents_Mass");
  TH1F * resonanceHisto = (TH1F*)resonanceFile->Get("hRecBestRes_Mass");
  TH1F * backgroundHisto = (TH1F*)backgroundFile->Get("hRecBestRes_Mass");

  int rebinCount = 4;

  allHisto->Rebin(rebinCount);
  resonanceHisto->Rebin(rebinCount);
  backgroundHisto->Rebin(rebinCount);

  double xMin = allHisto->GetXaxis()->GetXmin();
  double xMax = allHisto->GetXaxis()->GetXmax();
  double deltaX = xMax - xMin;

  // The fitted background function gives the background fraction (is normalized).
  // We multiply it by the integral to get the background value.
  int xBins = allHisto->GetNbinsX();

  double OriginalResMass[] = {91.1876, 10.3552, 10.0233, 9.4603, 3.68609, 3.096916};
  double ResMass[] = {91.1876, 0., 0., (10.3552 + 10.0233 + 9.4603)/3., (3.68609 + 3.096916)/2., (3.68609 + 3.096916)/2.};
  double ResHalfWidth[] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};
  TString ResName[] = {"Z", "Upsilon3S", "Upsilon2S", "Upsilon1S", "Psi2S", "J/Psi"};

  std::vector<int> ires;
  std::vector<double> Bgrp1;
  std::vector<double> a;
  std::vector<double> leftWindowBorder;
  std::vector<double> rightWindowBorder;
  int backgroundType = 0;

  // IMPORTANT: parameters to change
  // -------------------------------
  ires.push_back(5);

  // Exponential
  backgroundType = 0;
  Bgrp1.push_back(0.730566);
  a.push_back(0.826053);
  a.push_back(0);
  leftWindowBorder.push_back(2.5);
  rightWindowBorder.push_back(5.);

  // // Linear
  // backgroundType.push_back(1);
  // Bgrp1.push_back(0.833);
  // a.push_back(0.00016);
  // a.push_back(-1.27329e-11);
  // leftWindowBorder.push_back(1);
  // rightWindowBorder.push_back(1);

  // // Atan
  // backgroundType.push_back(2);
  // Bgrp1.push_back(0.000547261);
  // a.push_back(10.0145);
  // a.push_back(-0.320232);
  // leftWindowBorder.push_back(1);
  // rightWindowBorder.push_back(1);

  // -------------------------------

  // Create histograms for the background functions
  std::vector<TH1F*> backgroundFunctionHisto;
  for( unsigned int i=0; i<ires.size(); ++i ) {
    backgroundFunctionHisto.push_back( buildHistogram(ResMass, ResHalfWidth, xBins, deltaX, xMin, xMax,
                                                      ires[i], Bgrp1[i], a[i], leftWindowBorder[i], rightWindowBorder[i],
						      allHisto, backgroundType, a[i+1]) );
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
    if( i == 1 ) backgroundFunctionHisto[i]->SetLineColor(kRed);
    else backgroundFunctionHisto[i]->SetLineColor(kGreen);

    backgroundFunctionHisto[i]->Draw("same");

    TH1F * whiteHisto = subRangeHisto(OriginalResMass, ResHalfWidth, ires[i], backgroundFunctionHisto[i], "whiteHisto");
    whiteHisto->SetLineWidth(2);
    whiteHisto->SetLineColor(kWhite);
    whiteHisto->Draw("same");

    TH1F * dashedHisto = subRangeHisto(OriginalResMass, ResHalfWidth, ires[i], backgroundFunctionHisto[i], "dashedHisto");
    dashedHisto->SetLineWidth(2);
    dashedHisto->SetLineStyle(7);
    dashedHisto->Draw("same");

    legend->AddEntry(backgroundFunctionHisto[i], "Background function for "+ResName[ires[i]]);
  }

  legend->Draw("same");

  canvas->GetPad(0)->SetLogy(true);

  canvas->Print("BackgroundCheck.pdf");
}

TH1F * buildHistogram(const double * ResMass, const double * ResHalfWidth, const int xBins,
		      const double & deltaX, const double & xMin, const double & xMax,
                      const int ires, const double & Bgrp1, const double & a,
		      const double & leftWindowBorder, const double & rightWindowBorder,
		      const TH1F* allHisto, const int backgroundType, const double & b)
{
  // For J/Psi exclude the Upsilon from the background normalization as the bin is not used by the fit.
  double lowWindowValue = leftWindowBorder;
  double upWindowValue = rightWindowBorder;

  int lowBin = int((lowWindowValue)*xBins/deltaX);
  int upBin = int((upWindowValue)*xBins/deltaX);

  std::cout << "lowBin = " << lowBin << ", upBin = " << upBin << std::endl;
  std::cout << "lowWindowValue = " << lowWindowValue << ", upWindowValue = " << upWindowValue << std::endl;

  double xWidth = deltaX/xBins;


  TF1 * backgroundFunction = 0;
  TH1F * backgroundFunctionHisto = 0;

  if( backgroundType == 0 ) {
    // Exponential
    // -----------
    // backgroundFunction = new TF1("backgroundFunction", "[0]*([1]*exp(-[1]*x))", xMin, xMax );
    std::stringstream ssUp;
    std::stringstream ssDown;
    ssUp << upWindowValue;
    ssDown << lowWindowValue;
    std::string functionString("[0]*(-[1]*exp(-[1]*x)/(exp(-[1]*("+ssUp.str()+")) - exp(-[1]*("+ssDown.str()+")) ))");

    backgroundFunction = new TF1("backgroundFunction", functionString.c_str(), xMin, xMax );
    backgroundFunction->SetParameter(0, 1);
    backgroundFunction->SetParameter(1, a);
    backgroundFunctionHisto = new TH1F("backgroundFunctionHisto", "backgroundFunctionHisto", xBins, xMin, xMax);
    for( int xBin = 0; xBin < xBins; ++xBin ) {
      backgroundFunctionHisto->SetBinContent(xBin+1, backgroundFunction->Integral(xBin*xWidth, (xBin+1)*xWidth));
    }
  }
  else if( backgroundType == 2 ) {

    std::stringstream ssUp;
    std::stringstream ssDown;
    ssUp << upWindowValue;
    ssDown << lowWindowValue;

    Atan * atanFunction = new Atan(lowWindowValue, upWindowValue);
    backgroundFunction = new TF1("backgroundFunction", atanFunction, 0, 1, 3, "atanFunction");

    backgroundFunction->SetParameter(0, 1);
    backgroundFunction->SetParameter(1, a);
    backgroundFunction->SetParameter(2, b);
    backgroundFunctionHisto = new TH1F("backgroundFunctionHisto", "backgroundFunctionHisto", xBins, xMin, xMax);
    for( int xBin = 0; xBin < xBins; ++xBin ) {
      backgroundFunctionHisto->SetBinContent(xBin+1, backgroundFunction->Integral(xBin*xWidth, (xBin+1)*xWidth));
    }
  }
  else {
    // Linear
    // ------
    Linear * linearFunction = new Linear(lowWindowValue, upWindowValue);
    backgroundFunction = new TF1("backgroundFunction", linearFunction, 0, 1, 3, "linearFunction");
    backgroundFunction->SetParameter(0, 1);
    backgroundFunction->SetParameter(1, a);
    backgroundFunction->SetParameter(2, b);
    backgroundFunctionHisto = new TH1F("backgroundFunctionHisto", "backgroundFunctionHisto", xBins, xMin, xMax);
    for( int xBin = 0; xBin < xBins; ++xBin ) {
      backgroundFunctionHisto->SetBinContent(xBin+1, backgroundFunction->Integral(xBin*xWidth, (xBin+1)*xWidth));
    }
  }

  // The integral of the background function is 1
  // The function must be rescaled multiplying it to the number of events and k = Bgrp1
  double totEvents = allHisto->Integral(lowBin, upBin);
  backgroundFunctionHisto->Scale(Bgrp1*totEvents);
  backgroundFunction->SetParameter(0, Bgrp1*totEvents);

  std::cout << "Total events in the background window = " << totEvents << std::endl;
  std::cout << "FunctionHisto integral = " << backgroundFunctionHisto->Integral(lowBin, upBin) << std::endl;
  std::cout << "Function integral = " << backgroundFunction->Integral(lowWindowValue, upWindowValue) << std::endl;

  backgroundFunctionHisto->SetAxisRange(lowWindowValue, upWindowValue);

  return backgroundFunctionHisto;
}
