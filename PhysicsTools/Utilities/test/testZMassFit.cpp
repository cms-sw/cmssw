#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/RootFunctionAdapter.h"
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
using namespace std;
using namespace boost;

int main() { 
  gROOT->SetStyle("Plain");
  function::Parameter yield("Yield", 10000);
  function::Parameter mass("Mass", 91.2);
  function::Parameter gamma("Gamma", 2.5);
  function::Parameter dyield("Yield Error", 0);
  function::Parameter dmass("Mass Error", 0);
  function::Parameter dgamma("Gamma Error", 0);
  function::BreitWigner bw(mass, gamma);
  function::Constant c(yield);
  typedef function::Product<function::Constant, function::BreitWigner> FitFunction;
  FitFunction f = c * bw;
  TF1 fun = root::tf1("fun", f, 0, 200, yield, mass, gamma);
  TH1D histo("histo", "Z mass (GeV/c)", 200, 0, 200);
  histo.FillRandom("fun", yield);
  TCanvas canvas;
  fun.Draw();
  canvas.SaveAs("breitWigned.eps");
  histo.Draw();
  canvas.SaveAs("breitWignedHisto.eps");
  fun.Draw("same");
  canvas.SaveAs("breitWignedHistoFun.eps");
  histo.Draw("e");
  fun.Draw("same");
  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
  ChiSquared chi2(f, &histo, 80, 120);
  int fullBins = chi2.degreesOfFreedom();
  cout << "N. deg. of freedom: " << fullBins << endl;
  fit::RootMinuit<ChiSquared> minuit(3, chi2, true);
  minuit.setParameter(0, yield, 10, 100, 100000);
  minuit.setParameter(1, mass, .1, 70., 110);
  minuit.setParameter(2, gamma, 1, 1, 10);
  double amin = minuit.minimize();
  cout << "fullBins = " << fullBins 
       << "; free pars = " << minuit.getNumberOfFreeParameters() 
       << endl;
  unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
  cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
    //       << "; prob: " << TMath::Prob( amin, ndof )
       << endl;
  yield = minuit.getParameter(0);
  dyield = minuit.getParameterError(0);
  cout << yield << " ; " << dyield << endl;
  mass = minuit.getParameter(1);
  dmass = minuit.getParameterError(1);
  cout << mass << " ; " << dmass << endl;
  gamma = minuit.getParameter(2);
  dgamma = minuit.getParameterError(2);
  cout << gamma << " ; " << dgamma << endl;
  fun.SetParameters(yield, mass, gamma);
  fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str());
  fun.SetLineColor(kRed);
  fun.Draw("same");
  canvas.SaveAs("breitWignedHistoFunFit.eps");
  return 0;
}
