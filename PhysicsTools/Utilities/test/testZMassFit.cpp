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
  typedef HistoChiSquare<FitFunction> ChiSquared;
  ChiSquared chi2(f, &histo, histo.GetNbinsX(), 80, 120);
  int ndof = chi2.degreesOfFreedom();
  cout << "N. deg. of freedom: " << ndof << endl;
  fit::RootMinuit<ChiSquared> minuit(3, chi2, true);
  minuit.setParameter(0, yield, 10, 100, 100000);
  minuit.setParameter(1, mass, .1, 70., 110);
  minuit.setParameter(2, gamma, 1, 1, 10);
  //double amin =
  minuit.minimize();
  return 0;
}
