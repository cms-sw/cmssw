#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include "TFile.h"
#include "TH1.h"
#include <string>
#include <boost/shared_ptr.hpp>
using namespace std;
using namespace boost;

int main() { 
  TFile * ZToLL_file1 = new TFile("zMass.root","read");
  TH1D * zMass = (TH1D*) ZToLL_file1->Get("zMass");
  string names[] = { "P0", "mZmm", "GZmm" };
  shared_ptr<double> 
    P0(new double(6000)), 
    mZmm(new double(91.3)), 
    GZmm(new double(2.5));
  BreitWigner bw(mZmm, GZmm);
  Polynomial<0> c(P0);
  typedef Product<Polynomial<0>, BreitWigner> FitFunction;
  FitFunction f(c, bw);
  typedef HistoChiSquare<FitFunction> ChiSquared;
  ChiSquared chi2(f, zMass, zMass->GetNbinsX(), 80, 120);
  //  int fullBins = chi2.degreesOfFreedom();
  fit::RootMinuit<ChiSquared> rMinuit(3, chi2, true);
  rMinuit.setParameter(0, names[0].c_str(), P0, 10, 100, 100000);
  rMinuit.setParameter(1, names[1].c_str(), mZmm, .1, 70., 110);
  rMinuit.setParameter(2, names[2].c_str(), GZmm, 1, 1, 10);
  /*double amin = */ rMinuit.minimize();
  
  return 0;
}
