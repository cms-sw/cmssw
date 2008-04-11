#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/rootTf1.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
#include "PhysicsTools/Utilities/interface/Operations.h"
//using namespace std;
//using namespace boost;

int main() { 
  gROOT->SetStyle("Plain");
  typedef funct::Product<funct::Parameter, funct::BreitWigner>::type FitFunction;
  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
  try {
    fit::RootMinuitCommands<ChiSquared> commands("PhysicsTools/Utilities/test/testZMassFit.txt");
    
    const char * kYield = "Yield";
    const char * kMass = "Mass";
    const char * kGamma = "Gamma";
    
    funct::Parameter yield(kYield, commands.par(kYield));
    funct::Parameter mass(kMass, commands.par(kMass));
    funct::Parameter gamma(kGamma, commands.par(kGamma));
    funct::BreitWigner bw(mass, gamma);
    
    FitFunction f = yield * bw;
    TF1 startFun = root::tf1("startFun", f, 0, 200, yield, mass, gamma);
    TH1D histo("histo", "Z mass (GeV/c)", 200, 0, 200);
    histo.FillRandom("startFun", yield);
    TCanvas canvas;
    startFun.Draw();
    canvas.SaveAs("breitWigner.eps");
    histo.Draw();
    canvas.SaveAs("breitWignerHisto.eps");
    startFun.Draw("same");
    canvas.SaveAs("breitWignerHistoFun.eps");
    histo.Draw("e");
    startFun.Draw("same");
    
    ChiSquared chi2(f, &histo, 80, 120);
    int fullBins = chi2.degreesOfFreedom();
    std::cout << "N. deg. of freedom: " << fullBins << std::endl;
    fit::RootMinuit<ChiSquared> minuit(chi2, true);
    commands.add(minuit, yield);
    commands.add(minuit, mass);
    commands.add(minuit, gamma);
    commands.run(minuit);
    root::plot<FitFunction>("breitWignerHistoFunFit.eps", histo, f, 80, 120, yield, mass, gamma);
  } catch(std::exception & err){
    std::cerr << "Exception caught:\n" << err.what() << std::endl;
    return 1;
  }
  
  return 0;
}
