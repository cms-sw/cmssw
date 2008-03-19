#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
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
  typedef function::Product<function::Constant, function::BreitWigner> FitFunction;
  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
  try {
    fit::RootMinuitCommands<ChiSquared> commands("PhysicsTools/Utilities/test/testZMassFit.txt");
    
    const char * kYield = "Yield";
    const char * kMass = "Mass";
    const char * kGamma = "Gamma";
    
    function::Parameter yield(kYield, commands.par(kYield));
    function::Parameter mass(kMass, commands.par(kMass));
    function::Parameter gamma(kGamma, commands.par(kGamma));
    function::BreitWigner bw(mass, gamma);
    function::Constant c(yield);
    
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
    
    ChiSquared chi2(f, &histo, 80, 120);
    int fullBins = chi2.degreesOfFreedom();
    cout << "N. deg. of freedom: " << fullBins << endl;
    fit::RootMinuit<ChiSquared> minuit(chi2, true);
    commands.add(minuit, yield);
    commands.add(minuit, mass);
    commands.add(minuit, gamma);
    commands.run(minuit);
    fun.SetParameters(yield, mass, gamma);
    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str());
    fun.SetLineColor(kRed);
    fun.Draw("same");
    canvas.SaveAs("breitWignedHistoFunFit.eps");
  } catch(std::exception & err){
    cerr << "Exception caught:\n" << err.what() << endl;
    return 1;
  }
  
  return 0;
}
