#include "PhysicsTools/Utilities/interface/Likelihood.h"
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
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

#include <iostream>
#include "PhysicsTools/Utilities/interface/Operations.h"
//using namespace std;
//using namespace boost;

int main() {
  gROOT->SetStyle("Plain");
  typedef funct::BreitWigner PDF;
  typedef std::vector<double> Sample;
  typedef fit::Likelihood<Sample, PDF> Likelihood;
  typedef funct::Product<funct::Parameter, PDF>::type PlotFunction;
  try {
    fit::RootMinuitCommands<Likelihood> commands("PhysicsTools/Utilities/test/testZMassFitLikelihood.txt");

    const char* kYield = "Yield";
    const char* kMass = "Mass";
    const char* kGamma = "Gamma";

    funct::Parameter yield(kYield, commands.par(kYield));
    funct::Parameter mass(kMass, commands.par(kMass));
    funct::Parameter gamma(kGamma, commands.par(kGamma));
    funct::BreitWigner bw(mass, gamma);

    PDF pdf = bw;
    PlotFunction f = yield * pdf;
    TF1 startFun = root::tf1("startFun", f, 0, 200, yield, mass, gamma);
    TH1D histo("histo", "Z mass (GeV/c)", 200, 0, 200);
    Sample sample;
    sample.reserve(yield);
    for (unsigned int i = 0; i < yield; ++i) {
      double m = startFun.GetRandom();
      histo.Fill(m);
      sample.push_back(m);
    }
    TCanvas canvas;
    startFun.Draw();
    canvas.SaveAs("breitWigner.eps");
    histo.Draw();
    canvas.SaveAs("breitWignerHisto.eps");
    startFun.Draw("same");
    canvas.SaveAs("breitWignerHistoFun.eps");
    histo.Draw("e");
    startFun.Draw("same");

    Likelihood like(sample, pdf);
    fit::RootMinuit<Likelihood> minuit(like, true);
    commands.add(minuit, mass);
    commands.add(minuit, gamma);
    commands.run(minuit);
    ROOT::Math::SMatrix<double, 2, 2, ROOT::Math::MatRepSym<double, 2> > err;
    minuit.getErrorMatrix(err);
    std::cout << "error matrix:" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        std::cout << err(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    root::plot<PlotFunction>("breitWignerHistoFunFit.eps", histo, f, 80, 120, yield, mass, gamma);
  } catch (std::exception& err) {
    std::cerr << "Exception caught:\n" << err.what() << std::endl;
    return 1;
  }

  return 0;
}
