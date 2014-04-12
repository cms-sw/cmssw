#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TF1.h"
#include <cmath>
#include <sstream>
#include "TROOT.h"
#include "TStyle.h"
#include "TMath.h"

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
  TF1 * functionToFit = new TF1("functionToFit", crystalBall, 60, 120, 5);
  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  functionToFit->SetParameter(2, 1.);
  functionToFit->SetParameter(3, 1.);
  functionToFit->SetParameter(4, 1.);
  return functionToFit;
}

// Double_t reversedCrystalBall(Double_t *x, Double_t *par)
// {
//   double x_0 = par[0];
//   double sigma = par[1];
//   double n = par[2];
//   double alpha = par[3];
//   double N = par[4];

//   double A = pow(n/fabs(alpha), n)*exp(-(alpha*alpha)/2);
//   double B = n/fabs(alpha) - fabs(alpha);

//   if((x[0]-x_0)/sigma < alpha) {
//     return N*exp(-(x[0]-x_0)*(x[0]-x_0)/(2*sigma*sigma));
//   }
//   else {
//     return N*A*pow( (B + (x[0]-x_0)/sigma), -n );
//   }
// }
// TF1 * reversedCrystalBallFit()
// {
//   TF1 * functionToFit = new TF1("functionToFit", reversedCrystalBall, 60, 120, 5);
//   functionToFit->SetParameter(0, 90);
//   functionToFit->SetParameter(1, 10);
//   functionToFit->SetParameter(2, 1.);
//   functionToFit->SetParameter(3, 1.);
//   functionToFit->SetParameter(4, 1.);
//   return functionToFit;
// }

// Lorentzian function and fit (non relativistic Breit-Wigner)
// -----------------------------------------------------------
TF1 * lorentzian = new TF1( "lorentzian", "[2]/((x-[0])*(x-[0])+(([1]/2)*([1]/2)))", 0, 1000);
TF1 * lorentzianFit()
{
  TF1 * functionToFit = new TF1("functionToFit", lorentzian, 60, 120, 3);
  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  return functionToFit;
}

// Relativistic Breit-Wigner
// -------------------------
TF1 * relativisticBW = new TF1 ("relativisticBW", "[2]*pow([0]*[1],2)/(pow(x*x-[1]*[1],2)+pow([0]*[1],2))", 0., 1000.);
TF1 * relativisticBWFit(const std::string & index)
{
  TF1 * functionToFit = new TF1(("functionToFit"+index).c_str(), relativisticBW, 60, 120, 3);
  functionToFit->SetParameter(1, 90);
  functionToFit->SetParameter(0, 2);
  return functionToFit;
}

// Relativistic Breit-Wigner with Z/gamma interference term
// --------------------------------------------------------
class RelativisticBWwithZGammaInterference
{
 public:
  RelativisticBWwithZGammaInterference()
  {
    parNum_ = 4;
    twoOverPi_ = 2./TMath::Pi();
  }
  double operator() (double *x, double *p)
  {
    double squaredMassDiff = pow((x[0]*x[0] - p[1]*p[1]), 2);
    double denominator = squaredMassDiff + pow(x[0], 4)*pow(p[0]/p[1], 2);
    return p[2]*( p[3]*twoOverPi_*pow(p[1]*p[0], 2)/denominator + (1-p[3])*p[1]*squaredMassDiff/denominator );
  }
  int parNum() const { return parNum_; }
 protected:
  int parNum_;
  double twoOverPi_;
};

TF1 * relativisticBWintFit(const std::string & index)
{
  RelativisticBWwithZGammaInterference * fobj = new RelativisticBWwithZGammaInterference;
  TF1 * functionToFit = new TF1(("functionToFit"+index).c_str(), fobj, 60, 120, fobj->parNum(), "RelativisticBWwithZGammaInterference");
  functionToFit->SetParameter(0, 2.);
  functionToFit->SetParameter(1, 90.);
  functionToFit->SetParameter(2, 1.);
  functionToFit->SetParameter(3, 1.);

  functionToFit->SetParLimits(3, 0., 1.);

  return functionToFit;
}




// Relativistic Breit-Wigner with Z/gamma interference term and photon propagator
// ------------------------------------------------------------------------------
class RelativisticBWwithZGammaInterferenceAndPhotonPropagator
{
 public:
  RelativisticBWwithZGammaInterferenceAndPhotonPropagator()
  {
    parNum_ = 5;
    twoOverPi_ = 2./TMath::Pi();
  }
  double operator() (double *x, double *p)
  {

    // if( p[3]+p[4] > 1 ) return -10000.;

    double squaredMassDiff = pow((x[0]*x[0] - p[1]*p[1]), 2);
    double denominator = squaredMassDiff + pow(x[0], 4)*pow(p[0]/p[1], 2);
    return p[2]*( p[3]*twoOverPi_*pow(p[1]*p[0], 2)/denominator + (1-p[3]-p[4])*p[1]*squaredMassDiff/denominator + p[4]/(x[0]*x[0]));
  }
  int parNum() const { return parNum_; }
 protected:
  int parNum_;
  double twoOverPi_;
};

TF1 * relativisticBWintPhotFit(const std::string & index)
{
  RelativisticBWwithZGammaInterferenceAndPhotonPropagator * fobj = new RelativisticBWwithZGammaInterferenceAndPhotonPropagator;
  TF1 * functionToFit = new TF1(("functionToFit"+index).c_str(), fobj, 60, 120, fobj->parNum(), "RelativisticBWwithZGammaInterferenceAndPhotonPropagator");
  functionToFit->SetParameter(0, 2.);
  functionToFit->SetParameter(1, 90.);
  functionToFit->SetParameter(2, 1.);
  functionToFit->SetParameter(3, 1.);
  functionToFit->SetParameter(4, 0.);

  functionToFit->SetParLimits(3, 0., 1.);
  functionToFit->SetParLimits(4, 0., 1.);

  return functionToFit;
}






// Product between an exponential term and the relativistic Breit-Wigner with Z/gamma interference term and photon propagator
// --------------------------------------------------------------------------------------------------------------------------
class ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator
{
 public:
  ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator()
  {
    parNum_ = 6;
    twoOverPi_ = 2./TMath::Pi();
  }
  double operator() (double *x, double *p)
  {

    // if( p[3]+p[4] > 1 ) return -10000.;

    double squaredMassDiff = pow((x[0]*x[0] - p[1]*p[1]), 2);
    double denominator = squaredMassDiff + pow(x[0], 4)*pow(p[0]/p[1], 2);
    return p[2]*exp(-p[5]*x[0])*( p[3]*twoOverPi_*pow(p[1]*p[0], 2)/denominator + (1-p[3]-p[4])*p[1]*squaredMassDiff/denominator + p[4]/(x[0]*x[0]));
  }
  int parNum() const { return parNum_; }
 protected:
  int parNum_;
  double twoOverPi_;
};

TF1 * expRelativisticBWintPhotFit(const std::string & index)
{
  ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator * fobj = new ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator;
  TF1 * functionToFit = new TF1(("functionToFit"+index).c_str(), fobj, 60, 120, fobj->parNum(), "ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator");
  functionToFit->SetParameter(0, 2.);
  functionToFit->SetParameter(1, 90.);
  functionToFit->SetParameter(2, 1.);
  functionToFit->SetParameter(3, 1.);
  functionToFit->SetParameter(4, 0.);
  functionToFit->SetParameter(5, 0.);

  functionToFit->SetParLimits(3, 0., 1.);
  functionToFit->SetParLimits(4, 0., 1.);

  return functionToFit;
}







// Residual fit
// ------------
class ResidualFit
{
 public:
  ResidualFit()
  {
    parNum_ = 6;
  }
  double operator() (double *x, double *p)
  {
//     return p[0] + p[1]*x[0] + p[2]*x[0]*x[0] + p[3]/x[0] + p[4]/(x[0]*x[0]) + p[5]/(pow(x[0],3)) + p[6]*pow(x[0],p[7]);

    // if( x[0] < 90.25 ) return (p[0] + p[1]/x[0] + p[2]*x[0] + p[3]*x[0]*x[0] + p[4]*pow(x[0],3))/(p[5] + p[6]/x[0] + p[7]*x[0] + p[8]*x[0]*x[0] + p[9]*pow(x[0],3));
    if( x[0] < 89.5 ) return p[0] + p[1]*x[0] + p[2]*x[0]*x[0] + p[3]/x[0] + p[4]/(x[0]*x[0]) + p[5]/(pow(x[0],3));
    // if( x[0] < 92 ) return p[6] + p[7]/x[0];
    if( x[0] < 90.56 ) return p[6] + p[7]*x[0] + p[8]*x[0]*x[0];
    // if( x[0] < 94.5 ) return p[8] + p[9]/x[0];
    if( x[0] < 91.76 ) return p[0] + p[1]*x[0] + p[2]*x[0]*x[0];
//     if( x[0] < 94 ) return p[8] + p[9]*x[0] + p[14]*x[0]*x[0];
//     return p[10] + p[11]*x[0] + p[12]*x[0]*x[0] + p[15]*pow(x[0],3) + p[16]*exp(p[17]*x[0]);
//     // else return p[10] + p[11]/x[0] + p[12]*x[0] + p[13]*x[0] + p[14]*x[0] + p[15]*exp(p[16]*x[0]);

    return p[3] + p[4]/x[0] + p[5]/pow(x[0], 2);

  }
  int parNum() const { return parNum_; }
 protected:
  int parNum_;
};

TF1 * residualFit(const std::string & index)
{
  ResidualFit * fobj = new ResidualFit;
  TF1 * functionToFit = new TF1(("f"+index).c_str(), fobj, 60, 120, fobj->parNum(), "ResidualFit");

  double pars[] = { -2.30972e-02, 7.94253e-01, 7.40701e-05, 2.79889e-06,
                    -2.04844e-08, -5.78370e-43, 6.84145e-02, -6.21316e+00,
                    -6.90792e-03, 6.43131e-01, -2.03049e-03, 3.70415e-05,
                    -1.69014e-07 };
  functionToFit->SetParameters(pars);

  return functionToFit;
}

TF1 * residualFit(double pars[], const std::string & index)
{
  ResidualFit * fobj = new ResidualFit;
  TF1 * functionToFit = new TF1(("fp"+index).c_str(), fobj, 60, 120, fobj->parNum(), "ResidualFit");

//   double pars[] = { -2.30972e-02, 7.94253e-01, 7.40701e-05, 2.79889e-06,
//                     -2.04844e-08, -5.78370e-43, 6.84145e-02, -6.21316e+00,
//                     -6.90792e-03, 6.43131e-01, -2.03049e-03, 3.70415e-05,
//                     -1.69014e-07 };
  functionToFit->SetParameters(pars);

  return functionToFit;
}





class CombinedFit
{
 public:
  CombinedFit(double pars[], const std::string & index)
  {
    residuals_ = residualFit(pars, index);

    parNum_ = 3;
  }
  double operator() (double *x, double *p)
  {
    return p[2]/((x[0]-p[0])*(x[0]-p[0])+((p[1]/2)*(p[1]/2))) + residuals_->Eval(x[0]);
    // return p[2]/((x[0]-p[0])*(x[0]-p[0])+((p[1]/2)*(p[1]/2)));
  }
  int parNum() const { return parNum_; }
 protected:
  int parNum_;
  TF1 * residuals_;
};

TF1 * combinedFit(double pars[], const std::string & index)
{
  CombinedFit * combined = new CombinedFit(pars, index);
  TF1 * functionToFit = new TF1(("functionToFit"+index).c_str(), combined, 60, 120, combined->parNum(), "functionToFit");


  functionToFit->SetParameter(0, 90);
  functionToFit->SetParameter(1, 10);
  functionToFit->SetParameter(2, 1.);

  return functionToFit;
}



TF1 * iterateFitter(TH1F * histo, int i, TF1 * previousResiduals = 0, TCanvas * inputCanvas = 0)
{
  TCanvas * canvas = inputCanvas;
  std::stringstream ss;
  int iCanvas = i;
  ss << i;

  int nBins = histo->GetNbinsX();

  TF1 * functionToFit;
  if( previousResiduals == 0 ) {
    // functionToFit = relativisticBWFit(ss.str());
    // functionToFit = relativisticBWintFit(ss.str());
    // functionToFit = relativisticBWintPhotFit(ss.str());
    functionToFit = expRelativisticBWintPhotFit(ss.str());
    // functionToFit = crystalBallFit();
    // functionToFit = reversedCrystalBallFit();
    // functionToFit = lorentzianFit();
    // functionToFit = relativisticBWFit();
  }
  else {
    functionToFit = combinedFit(previousResiduals->GetParameters(), ss.str());
  }
  histo->Fit(functionToFit, "MN", "", 60, 120);


  double xMin = histo->GetXaxis()->GetXmin();
  double xMax = histo->GetXaxis()->GetXmax();
  double step = (xMax-xMin)/(double(nBins));
  TH1F * functionHisto = new TH1F(("functionHisto"+ss.str()).c_str(), "functionHisto", nBins, xMin, xMax);
  for( int i=0; i<nBins; ++i ) {
    functionHisto->SetBinContent( i+1, functionToFit->Eval(xMin + (i+0.5)*step) );
  }

  if( canvas == 0 ) {
    canvas = new TCanvas(("canvasResiduals"+ss.str()).c_str(), ("canvasResiduals"+ss.str()).c_str(), 1000, 800);
    canvas->Divide(2,1);
    canvas->Draw();
    iCanvas = 0;
  }
  canvas->cd(1+2*iCanvas);
  histo->Draw();
  functionToFit->SetLineColor(kRed);
  functionToFit->Draw("same");
  // functionHisto->Draw("same");
  // functionHisto->SetLineColor(kGreen);
  canvas->cd(2+2*iCanvas);
  TH1F * residuals = (TH1F*)histo->Clone();
  residuals->Add(functionHisto, -1);
  residuals->SetName("Residuals");

  // TF1 * residualFit = new TF1("residualFit", "-[0] + [1]*x+sqrt( ([1]-1)*([1]-1)*x*x + [0]*[0] )", 0., 1000. );
  // TF1 * residualFit = new TF1("residualFit", "[0]*(x - [1])/([2]*x*x + [3]*x + [4])", 0., 1000. );
  TF1 * residualFitFunction = residualFit(ss.str());
  residuals->Fit(residualFitFunction, "ME", "", 90.56, 120);

  residuals->Draw();

  return residualFitFunction;
}

void ProbsFitter()
{
  TFile * inputFile = new TFile("Sherpa_nocuts.root", "READ");
  TH1F * histo = (TH1F*)inputFile->FindObjectAny("HistAllZMass");
  histo->Scale(1/histo->GetEntries());
  histo->Rebin(6);
  TCanvas * canvas = new TCanvas("canvas", "canvas", 1000, 800);
  canvas->Divide(2,3);

  // gStyle->SetOptFit(1);

  TF1 * residuals1 = iterateFitter(histo, 0, 0, canvas);
//   TF1 * residuals2 = iterateFitter(histo, 1, residuals1, canvas);
//   iterateFitter(histo, 2, residuals2, canvas);

//   canvas->cd(3);
//   TH1F * hclone = (TH1F*)histo->Clone();
//   TF1 * combinedFunctionToFit = combinedFit(residualFitFunction->GetParameters());
//   hclone->Fit(combinedFunctionToFit, "MN", "", 60, 120);
//   hclone->Draw();
//   combinedFunctionToFit->Draw("same");
//   combinedFunctionToFit->SetLineColor(kRed);

}
