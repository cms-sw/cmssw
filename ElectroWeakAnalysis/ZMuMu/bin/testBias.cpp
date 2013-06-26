#include "TROOT.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/rootTf1.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "TH1.h"
#include "TF1.h"
#include "TFile.h"
#include "TRandom3.h"
#include <iostream>
using namespace std;

struct sig_tag { };
struct bkg1_tag{ };
struct bkg2_tag{ };

int main() {
  gROOT->SetStyle("Plain");
  typedef funct::FunctExpression Expr;
  typedef fit::MultiHistoChiSquare<Expr, Expr> ChiSquared;
  TRandom3 rndm;
  TFile file("out.root", "RECREATE");
  TH1F hSigPull("sigPull", "sig - pull", 100, -10, 10);
  TH1F hEffPull("effPull", "eff - pull", 100, -10, 10);
  TH1F hBkg1Pull("bkg1Pull", "bkg1 - pull", 100, -10, 10);
  TH1F hBkg2Pull("bkg2Pull", "bkg1 - pull", 100, -10, 10);
  bool firstTime = true;
  for(unsigned int n=0; n < 1000; ++n) {
    const char * kSig = "Sig";
    const char * kEff = "Eff";
    const char * kBkg1 = "Bkg1";
    const char * kBkg2 = "Bkg2";
    const char * kMass = "Mass";
    const char * kGamma = "Gamma";
    const char * kLambda1 = "Lambda1";
    const char * kLambda2 = "Lambda2";
    double sig_true = 1000;
    double eff_true = 0.95;
    double bkg1_true = 100;
    double bkg2_true = 40;
    double mass_true = 91.2;
    double gamma_true = 2.50;
    double lambda1_true = -0.01;
    double lambda2_true = -0.005;
    
    funct::Parameter sig(kSig, sig_true);
    funct::Parameter eff(kEff, eff_true);
    funct::Parameter bkg1(kBkg1, bkg1_true);
    funct::Parameter bkg2(kBkg2, bkg2_true);
    funct::Parameter mass(kMass, mass_true);
    funct::Parameter gamma(kGamma, gamma_true);
    funct::Parameter lambda1(kLambda1, lambda1_true);
    funct::Parameter lambda2(kLambda2, lambda2_true);
    const double n_rebin = 0.5;
    funct::Parameter rebin("rebin", n_rebin);

    funct::Numerical<2> _2;
    funct::Numerical<1> _1;
    funct::BreitWigner bw(mass, gamma);
    funct::Exponential expo1(lambda1);
    funct::Exponential expo2(lambda2);
    Expr fSig = sig * bw;
    Expr fBkg1 = bkg1 * expo1;
    Expr fBkg2 = bkg2 * expo2;
    Expr f1 = rebin*(_2 * eff * (_1 - eff) * fSig + fBkg1);
    Expr f2 = rebin*((eff ^ _2) * fSig + fBkg2);
    TF1 funSig = root::tf1_t<sig_tag, Expr>("fSig", fSig, 0, 200, sig, mass, gamma);
    TF1 funBkg1 = root::tf1_t<bkg1_tag, Expr>("fBkg1", fBkg1, 0, 200, bkg1, lambda1);
    TF1 funBkg2 = root::tf1_t<bkg2_tag, Expr>("fBkg2", fBkg2, 0, 200, bkg2, lambda2);
    int bins = int(200. / n_rebin);
    TH1D histo1("histo1", "Z mass (GeV/c)", bins, 0, 200);
    TH1D histo2("histo2", "Z mass (GeV/c)", bins, 0, 200);
    double areaBkg1 = funBkg1.Integral(0,200);
    double areaBkg2 = funBkg2.Integral(0,200);
    histo1.FillRandom("fBkg1", int(rndm.Poisson(areaBkg1)));
    histo2.FillRandom("fBkg2", int(rndm.Poisson(areaBkg2)));
    double areaSig = funSig.Integral(0, 200);
    int nSig = int(rndm.Poisson(areaSig));
    for(int i = 0; i < nSig; ++i) {
      bool pass1 = rndm.Uniform() < eff_true;
      bool pass2 = rndm.Uniform() < eff_true;
      double x = funSig.GetRandom();
      if((pass1 && !pass2) || (!pass1 && pass2)) histo1.Fill(x);
      if(pass1 && pass2) histo2.Fill(x);
    }
 
    if(firstTime) {
      histo1.Write();
      histo2.Write();
    }
    ChiSquared chi2(f1, &histo1, f2, &histo2, 80, 140);
    fit::RootMinuit<ChiSquared> minuit(chi2, true);
    minuit.addParameter(sig, 100, 0, 10000);
    minuit.addParameter(eff, 0.01, 0, 1);
    minuit.addParameter(mass, 2, 70, 120);
    minuit.addParameter(gamma, 1, 0, 5);
    minuit.addParameter(bkg1, 10, 0, 10000);
    minuit.addParameter(bkg2, 10, 0, 10000);
    minuit.addParameter(lambda1, 0.1, -5, 0);
    minuit.addParameter(lambda2, 0.1, -5, 0);
    minuit.minimize();
    minuit.migrad();
    
    double sigPull = (sig() - sig_true) / minuit.getParameterError(kSig);
    cout << "sig pull: " << sigPull << endl;
    double effPull = (eff() - eff_true) / minuit.getParameterError(kEff);
    cout << "eff pull: " << effPull << endl;
    double bkg1Pull = (bkg1() - bkg1_true) / minuit.getParameterError(kBkg1);
    cout << "bkg1Pull: " << bkg1Pull << endl;
    double bkg2Pull = (bkg2() - bkg2_true) / minuit.getParameterError(kBkg2);
    cout << "bkg2Pull: " << bkg2Pull << endl;
    hSigPull.Fill(sigPull);
    hEffPull.Fill(effPull);
    hBkg1Pull.Fill(bkg1Pull);
    hBkg2Pull.Fill(bkg2Pull);
    firstTime = false;
  }
  hSigPull.Write();
  hEffPull.Write();
  hBkg1Pull.Write();
  hBkg2Pull.Write();
  file.Close();
  return 0;
}
