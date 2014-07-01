#include <iostream>
#include "TH1F.h"
#include "TString.h"

/**
 * This class receives a TH1 histogram and applies a smoothing.
 * The smoothing can be done in two ways:
 * - with the applySmooth method:
 * -- for each bin take the two bins on the left and the right
 * -- compute the average of the sidepoints and compare with the middle
 * -- if the difference is more than N% (configurable) replace the point with the average
 * - with the applySmoothSingle method:
 * -- the same but using only the closest bin on left and right.
 */

class Smoother {
 public:
  Smoother( const int N = 0.5 ) : ratio_(N) {}

  TH1F * smooth( TH1F * histo, const int iterations, const bool * single, const double & newHistoXmin = 71.1876, const double & newHistoXmax = 111.1876, const int newHistoBins = 1001 );

 protected:
  TH1F * applySmooth( TH1F * histo, const double & newHistoXmin, const double & newHistoXmax, const int newHistoBins );
  TH1F * applySmoothSingle( TH1F * histo, const double & newHistoXmin, const double & newHistoXmax, const int newHistoBins );
  int ratio_;
};

TH1F * Smoother::applySmooth( TH1F * histo, const double & newHistoXmin, const double & newHistoXmax, const int newHistoBins )
{
  TH1F * smoothedHisto = (TH1F*)histo->Clone();
  smoothedHisto->Reset();

  histo->SetAxisRange(newHistoXmin, newHistoXmax);
  int xBinMin = histo->FindBin(newHistoXmin);
  int xBinMax = histo->FindBin(newHistoXmax);
  std::cout << "xBinMin = " << xBinMin << std::endl;
  std::cout << "xBinMax = " << xBinMax << std::endl;

  double xMin = histo->GetXaxis()->GetXmin();
  double xMax = histo->GetXaxis()->GetXmax();
  int nBins = histo->GetNbinsX();
  std::cout << "xMin = " << xMin << std::endl;
  std::cout << "xMax = " << xMax << std::endl;
  std::cout << "nBins = " << nBins << std::endl;

  // Start from the third bin up to the second to last bin
  int smoothedBin = 2;
  smoothedHisto->SetBinContent(0, histo->GetBinContent(0));
  smoothedHisto->SetBinContent(1, histo->GetBinContent(1));
  for( int i=2+xBinMin; i<xBinMax-2; ++i, ++smoothedBin ) {
    double min2 = histo->GetBinContent(i-2);
    double min1 = histo->GetBinContent(i-1);
    double med = histo->GetBinContent(i);
    double plus1 = histo->GetBinContent(i+1);
    double plus2 = histo->GetBinContent(i+2);
    // If the slope is the same before and after (med is not a min/max)
    if( (min1 - min2)*(plus2 - plus1) > 0 ) {
      // compare the med with the mean of the four points
      double newMed = ((min1+min2+plus1+plus2)/4);
      if( fabs(med/newMed - 1) > ratio_ ) {
	std::cout << "Replacing value for bin " << i << " from " << med << " to " << newMed << std::endl;
	// smoothedHisto->SetBinContent(smoothedBin, newMed);
	smoothedHisto->SetBinContent(i, newMed);
	continue;
      }
    }
    // smoothedHisto->SetBinContent(smoothedBin, med);
    smoothedHisto->SetBinContent(i, med);
    std::cout << "bin["<<i<<"] = " << histo->GetBinContent(i) << std::endl;
  }
  return smoothedHisto;
}

TH1F * Smoother::applySmoothSingle( TH1F * histo, const double & newHistoXmin, const double & newHistoXmax, const int newHistoBins )
{
  TH1F * smoothedHisto = (TH1F*)histo->Clone();
  smoothedHisto->Reset();

  histo->SetAxisRange(newHistoXmin, newHistoXmax);
  int xBinMin = histo->FindBin(newHistoXmin);
  int xBinMax = histo->FindBin(newHistoXmax);
  std::cout << "xBinMin = " << xBinMin << std::endl;
  std::cout << "xBinMax = " << xBinMax << std::endl;

  double xMin = histo->GetXaxis()->GetXmin();
  double xMax = histo->GetXaxis()->GetXmax();
  int nBins = histo->GetNbinsX();
  std::cout << "xMin = " << xMin << std::endl;
  std::cout << "xMax = " << xMax << std::endl;
  std::cout << "nBins = " << nBins << std::endl;

  // Start from the third bin up to the second to last bin
  int smoothedBin = 2;
  smoothedHisto->SetBinContent(0, histo->GetBinContent(0));
  smoothedHisto->SetBinContent(1, histo->GetBinContent(1));
  for( int i=2+xBinMin; i<xBinMax-2; ++i, ++smoothedBin ) {
    double min1 = histo->GetBinContent(i-1);
    double med = histo->GetBinContent(i);
    double plus1 = histo->GetBinContent(i+1);
    // compare the med with the mean of the four points
    double newMed = ((min1+plus1)/2);
    if( fabs(med/newMed - 1) > ratio_ ) {
      std::cout << "Replacing value for bin " << i << " from " << med << " to " << newMed << std::endl;
      smoothedHisto->SetBinContent(i, newMed);
      continue;
    }
    smoothedHisto->SetBinContent(i, med);
    std::cout << "bin["<<i<<"] = " << histo->GetBinContent(i) << std::endl;
  }
  return smoothedHisto;
}


TH1F * Smoother::smooth( TH1F * histo, const int iterations, const bool * single, const double & newHistoXmin, const double & newHistoXmax, const int newHistoBins )
{
  histo->Draw();
  TH1F * smoothedHisto = histo;
  for( int i=0; i<iterations; ++i ) {
    if( single[i] ) {
      smoothedHisto = applySmoothSingle(smoothedHisto, newHistoXmin, newHistoXmax, newHistoBins);
    }
    else {
      smoothedHisto = applySmooth(smoothedHisto, newHistoXmin, newHistoXmax, newHistoBins);
    }
//     smoothedHisto->Draw("same");
//     smoothedHisto->SetLineColor(2+i);
  }
  smoothedHisto->Draw("same");
  smoothedHisto->SetLineColor(kRed);
  return smoothedHisto;
}
