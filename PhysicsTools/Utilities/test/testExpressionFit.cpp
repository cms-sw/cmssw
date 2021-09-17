#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/rootTf1.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
#include "PhysicsTools/Utilities/interface/Function.h"
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TROOT.h"

#include <iostream>
#include "PhysicsTools/Utilities/interface/Operations.h"

int main() {
  typedef fit::HistoChiSquare<funct::Function<funct::X> > ChiSquared;
  gROOT->SetStyle("Plain");
  try {
    fit::RootMinuitCommands<ChiSquared> commands("PhysicsTools/Utilities/test/testExpressionFit.txt");

    const char* kYield = "Yield";
    const char* kMean = "Mean";
    const char* kSigma = "Sigma";

    funct::Parameter yield(kYield, commands.par(kYield));
    funct::Parameter mean(kMean, commands.par(kMean));
    funct::Parameter sigma(kSigma, commands.par(kSigma));
    funct::Parameter c("C", 1. / sqrt(2 * M_PI));
    funct::X x;
    funct::Numerical<2> _2;

    const double min = -5, max = 5;

    funct::Function<funct::X> f = yield * c * exp(-(((x - mean) / sigma) ^ _2) / _2);
    TF1 startFun = root::tf1("startFun", f, min, max, yield, mean, sigma);
    TH1D histo("histo", "gaussian", 100, min, max);
    histo.FillRandom("startFun", yield);
    TCanvas canvas;
    startFun.Draw();
    canvas.SaveAs("gaussian.eps");
    histo.Draw();
    canvas.SaveAs("gaussianHisto.eps");
    startFun.Draw("same");
    canvas.SaveAs("gaussianHistoFun.eps");
    histo.Draw("e");
    startFun.Draw("same");

    ChiSquared chi2(f, &histo, min, max);
    int fullBins = chi2.numberOfBins();
    std::cout << "N. deg. of freedom: " << fullBins << std::endl;
    fit::RootMinuit<ChiSquared> minuit(chi2, true);
    commands.add(minuit, yield);
    commands.add(minuit, mean);
    commands.add(minuit, sigma);
    commands.run(minuit);
    root::plot<funct::Function<funct::X> >("gaussianHistoFunFit.eps", histo, f, min, max, yield, mean, sigma);
  } catch (std::exception& err) {
    std::cerr << "Exception caught:\n" << err.what() << std::endl;
    return 1;
  }

  return 0;
}
