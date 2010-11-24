#include "HiggsAnalysis/CombinedLimit/interface/ProfileLikelihood.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"

using namespace RooStats;

bool ProfileLikelihood::run(RooWorkspace *w, RooAbsData &data, double &limit) {
  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);
  double rMax = r->getMax();
  for (;;) {
    ProfileLikelihoodCalculator plcB(data, *w->pdf("model_s"), poi);
    plcB.SetConfidenceLevel(0.95);
    LikelihoodInterval* plInterval = plcB.GetInterval();
    if (plInterval == 0) return false;
    limit = plInterval->UpperLimit(*r); 
    delete plInterval;
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit r < " << limit << "; r max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) return false;
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
      std::cout << "Limit: r < " << limit << " @ 95% CL" << std::endl;
      //std::cout << "Significance: " << plSig << std::endl;
      
    }
    break;
  }
  return true;
}
