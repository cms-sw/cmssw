#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/rootTf1.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "TH1.h"
#include "TF1.h"
#include <iostream>
#include "PhysicsTools/Utilities/interface/Operations.h"
//using namespace std;
//using namespace boost;

int main() { 
  typedef funct::Product<funct::Parameter, funct::BreitWigner>::type FitFunction;
  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
  try {
    funct::Parameter yield("Yield", 1000);
    funct::Parameter mass("Mass", 91.2);
    funct::Parameter gamma("Gamma", 2.50);
    funct::BreitWigner bw(mass, gamma);
    
    FitFunction f = yield * bw;
    TF1 startFun = root::tf1("startFun", f, 0, 200, yield, mass, gamma);
    TH1D histo("histo", "Z mass (GeV/c)", 200, 0, 200);
    histo.FillRandom("startFun", yield);
    ChiSquared chi2(f, &histo, 80, 120);
    fit::RootMinuit<ChiSquared> minuit(chi2, true);
    minuit.addParameter(yield, 100, 0, 10000);
    minuit.addParameter(mass, 2, 70, 120);
    minuit.addParameter(gamma, 1, 0, 5);
    minuit.minimize();
    minuit.migrad();
  } catch(std::exception & err){
    std::cerr << "Exception caught:\n" << err.what() << std::endl;
    return 1;
  }
  
  return 0;
}
