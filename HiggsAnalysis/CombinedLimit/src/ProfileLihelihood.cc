#include "HiggsAnalysis/CombinedLimit/interface/ProfileLikelihood.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"

#include <Math/MinimizerOptions.h>

using namespace RooStats;

ProfileLikelihood::ProfileLikelihood() :
    LimitAlgo("Profile Likelihood specific options")
{
    options_.add_options()
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value("Minuit2"), "Choice of minimizer (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(1e-3),  "Tolerance for minimizer")
    ;
}
bool ProfileLikelihood::run(RooWorkspace *w, RooAbsData &data, double &limit) {
  std::string minimizerTypeBackup = ROOT::Math::MinimizerOptions::DefaultMinimizerType();
  std::string minimizerAlgoBackup = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo();
  double      minimizerTollBackup = ROOT::Math::MinimizerOptions::DefaultTolerance();
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(minimizerTolerance_);
  if (minimizerAlgo_.find(",") != std::string::npos) {
      size_t idx = minimizerAlgo_.find(",");
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerAlgo_.substr(0,idx).c_str(), minimizerAlgo_.substr(idx+1).c_str());
  } else {
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerAlgo_.c_str());
  }

  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);
  double rMax = r->getMax();
  bool success = false;
  while (!success) {
    ProfileLikelihoodCalculator plcB(data, *w->pdf("model_s"), poi);
    plcB.SetConfidenceLevel(cl);
    LikelihoodInterval* plInterval = plcB.GetInterval();
    if (plInterval == 0) break;
    limit = plInterval->UpperLimit(*r); 
    delete plInterval;
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit r < " << limit << "; r max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) break;
      r->setMax(r->getMax()*2); 
      continue;
    }
    if (verbose) {
      /*
	ProfileLikelihoodCalculator plcS(data, *w->pdf("model_s"), poi);
	RooArgSet nullParamValues; 
	nullParamValues.addClone(*r); ((RooRealVar&)nullParamValues["r"]).setVal(0);
	plcS.SetNullParameters(nullParamValues);
	double plSig = plcS.GetHypoTest()->Significance();
      */
      
      std::cout << "\n -- Profile Likelihood -- " << "\n";
      std::cout << "Limit: r < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
      //std::cout << "Significance: " << plSig << std::endl;
      
    }
    success = true;
  }
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(minimizerTollBackup);
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerTypeBackup.c_str(),minimizerAlgoBackup.empty() ? 0 : minimizerAlgoBackup.c_str());
  return success;
}
