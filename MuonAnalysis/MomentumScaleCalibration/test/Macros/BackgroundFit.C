#include "TF1.h"
#include <cmath>
#include <iostream>
#include "TFile.h"
#include "TH1F.h"


/**
 * This macro can be used to fit the background and signal peak of the Z. <br>
 * It contains several different functions that can be activated by uncommenting
 * the BackgroundFit function.
 */

// e^-(t^2)/(y^2+(x-t)^2)
// x[0] = t
// par[0] = y
// par[1] = x
Double_t integralPart(Double_t *x, Double_t *par)
{
  std::cout << "x[0] = " << x[0] << std::endl;
  std::cout << "par[0] = " << par[0] << std::endl;
  std::cout << "par[1] = " << par[1] << std::endl;
  // std::cout << "value = " << exp(-(x[0]-par[1])*(x[0]-par[1])/(par[0]*par[0]))/(par[0]*par[0] + (par[1] - x[0])*(par[1] - x[0])) << std::endl;
  return exp(-x[0]*x[0])/(par[0]*par[0] + (par[1] - x[0])*(par[1] - x[0]));
}

// CrystalBall fit
// ---------------
// see for example http://en.wikipedia.org/wiki/Crystal_Ball_function
// par[0] = x0
// par[1] = sigma
// par[2] = A
// par[3] = B
// par[4] = alpha
// par[5] = N
// par[6] = n
Double_t crystalBall(Double_t *x, Double_t *par)
{
  double x_0 = par[0];
  double sigma = par[1];
  double n = par[2];
  double alpha = par[3];
  double N = par[4];

  double A = pow(n/fabs(alpha), n)*exp(-(alpha*alpha)/2);
  double B = n/fabs(alpha) - fabs(alpha);

  if((x[0]-x_0)/sigma > -alpha) {
    return N*exp(-(x[0]-x_0)*(x[0]-x_0)/(2*sigma*sigma));
  }
  else {
    return N*A*pow( (B - (x[0]-x_0)/sigma), -n );
  }
}
TF1 * crystalBallFit()
{
  TF1 * functionToFit = new TF1("functionToFit", crystalBall, 42, 160, 5);
  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  functionToFit->SetParameter(2, 1.);
  functionToFit->SetParameter(3, 1.);
  functionToFit->SetParameter(4, 1.);
  return functionToFit;
}

// Lorentzian function and fit
// ---------------------------
TF1 * lorentzian = new TF1( "lorentzian", "[2]/((x-[0])*(x-[0])+(([1]/2)*([1]/2)))", 0, 1000);
TF1 * lorentzianFit()
{
  TF1 * functionToFit = new TF1("functionToFit", lorentzian, 42, 160, 3);
  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  return functionToFit;
}

// Power law fit
// -------------
TF1 * powerLaw = new TF1( "powerLaw", "[0] + [1]*pow(x, [2])", 0, 1000 );
TF1 * powerLawFit()
{
  TF1 * functionToFit = new TF1("functionToFit", powerLaw, 42, 160, 3);
  return functionToFit;
}

// Lorentzian + and power law
TF1 * lorentzianAndPowerLaw()
{
  TF1 * functionToFit = new TF1("functionToFit", "[2]/((x-[0])*(x-[0])+(([1]/2)*([1]/2))) + ([3] + [4]*pow(x, [5]))", 42, 160);
  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  functionToFit->SetParameter(2, 0.33);
  functionToFit->SetParameter(3, -0.0220578);
  functionToFit->SetParameter(4, 0.0357716);
  functionToFit->SetParameter(5, -0.0962408);
  return functionToFit;
}

// Exponential fit
// ---------------
TF1 * exponential = new TF1 ("exponential", "[0]*exp([1]*x)", 0, 1000 );
TF1 * exponentialFit()
{
  TF1 * functionToFit = new TF1("functionToFit", exponential, 42, 160, 2);
  return functionToFit;
}

// Lorentzian + exponential fit
// ----------------------------
TF1 * lorenzianAndExponentialFit()
{
  TF1 * functionToFit = new TF1("functionToFit", "[2]/((x-[0])*(x-[0])+(([1]/2)*([1]/2))) + [3]*exp([4]*x)", 42, 160);
  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  functionToFit->SetParameter(2, 0.33);
  functionToFit->SetParameter(3, 0.0115272);
  functionToFit->SetParameter(4, -0.0294229);
  return functionToFit;
}



void BackgroundFit() {
  TFile* inputFile = new TFile("0_MuScleFit.root", "READ");
  TH1F* histo = (TH1F*)inputFile->FindObjectAny("hRecBestRes_Mass");
  histo->Rebin(30);
  histo->Scale(1/histo->GetEntries());

  // TF1 * functionToFit = lorentzianFit();
  // TF1 * functionToFit = crystalBallFit();
  // TF1 * functionToFit = powerLawFit();
  // TF1 * functionToFit = lorentzianAndPowerLaw();
  // TF1 * functionToFit = exponentialFit();
  TF1 * functionToFit = lorenzianAndExponentialFit();

  histo->Fit(functionToFit, "M", "", 42, 160);

}
