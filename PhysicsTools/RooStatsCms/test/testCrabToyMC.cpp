/*
 * Test tool for Toy Monte Carlo generation with CRAB submission
 *
 * \author Luca Lista, INFN
 *
 */
#include <boost/program_options.hpp>
#include "RooRandom.h"
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooGlobalFunc.h" 
#include "RooChi2Var.h"
#include "RooMinuit.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <string>

static const char * const kHelpOpt = "help";
static const char * const kHelpCommandOpt = "help,h";
static const char * const kSeedOpt = "seed";
static const char * const kSeedCommandOpt = "seed,s";

int main(int argc, char * argv[]) {
  using namespace boost::program_options;
  using namespace std;

  gROOT->SetBatch(kTRUE);
  gROOT->SetStyle("Plain");
  
  string programName(argv[0]);
  string descString(programName);
  descString += " [options] ";
  options_description desc(descString);

  desc.add_options()
    (kHelpCommandOpt, "produce help message")
    (kSeedCommandOpt, value<unsigned int>(), "random generator seed");

  positional_options_description p;

  p.add(kSeedOpt, -1);
  
  variables_map vm;
  try {
    store(command_line_parser(argc,argv).options(desc).positional(p).run(), vm);
    notify(vm);
  } catch(const error&) {
    return 7000;
  }

  if(vm.count(kHelpOpt)) {
    cout << desc <<std::endl;
    return 0;
  }

  unsigned int seed = 123456;
  if(vm.count(kSeedOpt)) {
    seed = vm[kSeedOpt].as<unsigned int>();
    cout << "random seed specified by user as: " << seed << endl;
  } else {
    cout << "random seed set by default as: " << seed << endl;
  }

  RooRandom::randomGenerator()->SetSeed(seed);

  RooRealVar x("x", "x", -10, 10);

  RooRealVar mu("mu", "average", 0, -1, 1);
  RooRealVar sigma("sigma", "sigma", 1, 0, 5);
  RooGaussian gauss("gauss","gaussian PDF", x, mu, sigma);

  RooRealVar lambda("lambda", "slope", -0.1, -5., 0.);
  RooExponential expo("expo", "exponential PDF", x, lambda);

  RooRealVar s("s", "signal yield", 1000, 0, 10000);
  RooRealVar b("b", "background yield", 1000, 0, 10000);

  cout << "initial values: " << endl;

  mu.Print();
  sigma.Print();
  lambda.Print();
  s.Print();
  b.Print();

  RooAddPdf sum("sum", "gaussian plus exponential PDF", 
		RooArgList(gauss, expo), RooArgList(s, b));

  unsigned int nSample = RooRandom::randomGenerator()->Poisson(s.getVal() + b.getVal());
  RooDataSet * data = sum.generate(x, nSample);
  x.setBins(50);
  RooDataHist hist("hist", "hist", RooArgSet(x), *data);
  RooChi2Var chi2("chi2","chi2", sum, hist, true);
  RooMinuit minuit(chi2);
  minuit.migrad();
  minuit.hesse();

  RooPlot * xFrame = x.frame() ;
  hist.plotOn(xFrame) ;
  sum.plotOn(xFrame) ;
  sum.plotOn(xFrame, RooFit::Components(expo), RooFit::LineStyle(kDashed)) ;
  TCanvas c;
  xFrame->Draw();
  c.SaveAs("binnedChi2Fit.eps");
  
  sum.getVariables()->Print();
  mu.Print();
  sigma.Print();
  lambda.Print();
  s.Print();
  b.Print();

  return 0;
}
 


