#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/HistoPoissonLikelihoodRatio.h"
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

typedef funct::Product<funct::Parameter, funct::BreitWigner>::type FitFunction;
typedef fit::HistoChiSquare<FitFunction> ChiSquared;
typedef fit::HistoPoissonLikelihoodRatio<FitFunction> PoissonLR;

template <typename T>
int main_t(const std::string tag) {
  try {
    fit::RootMinuitCommands<T> commands("PhysicsTools/Utilities/test/testZMassFit.txt");

    const char* kYield = "Yield";
    const char* kMass = "Mass";
    const char* kGamma = "Gamma";

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
    canvas.SaveAs((tag + "breitWigner.eps").c_str());
    histo.Draw();
    canvas.SaveAs((tag + "breitWignerHisto.eps").c_str());
    startFun.Draw("same");
    canvas.SaveAs((tag + "breitWignerHistoFun.eps").c_str());
    histo.Draw("e");
    startFun.Draw("same");

    T chi2(f, &histo, 80, 120);
    int fullBins = chi2.numberOfBins();
    std::cout << "N. deg. of freedom: " << fullBins << std::endl;
    fit::RootMinuit<T> minuit(chi2, true);
    commands.add(minuit, yield);
    commands.add(minuit, mass);
    commands.add(minuit, gamma);
    commands.run(minuit);
    ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3> > err;
    minuit.getErrorMatrix(err);
    std::cout << "error matrix:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        std::cout << err(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    root::plot<FitFunction>((tag + "breitWignerHistoFunFit.eps").c_str(), histo, f, 80, 120, yield, mass, gamma);
  } catch (std::exception& err) {
    std::cerr << "Exception caught:\n" << err.what() << std::endl;
    return 1;
  }

  return 0;
}

int main() {
  gROOT->SetStyle("Plain");
  std::cout << "=== chi-2 fit ===" << std::endl;
  int ret1 = main_t<ChiSquared>("chi2_");
  if (ret1 != 0)
    return ret1;
  std::cout << "=== poisson LR fit ===" << std::endl;
  int ret2 = main_t<PoissonLR>("possLR_");
  if (ret2 != 0)
    return ret2;
  return 0;
}
