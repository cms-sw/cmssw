#include "HiggsAnalysis/CombinedLimit/interface/BayesianFlatPrior.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooUniform.h"
#include "RooWorkspace.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"

using namespace RooStats;

bool BayesianFlatPrior::run(RooWorkspace *w, RooAbsData &data, double &limit) {
  RooRealVar *r = w->var("r");
  RooUniform  flatPrior("flatPrior","flatPrior",*r);
  RooArgSet  poi(*r);
  double rMax = r->getMax();
  for (;;) {
    BayesianCalculator bcalc(data, *w->pdf("model_s"), poi, flatPrior, (withSystematics_ ? w->set("nuisances") : 0));
    bcalc.SetLeftSideTailFraction(0);
    bcalc.SetConfidenceLevel(0.95); 
    SimpleInterval* bcInterval = bcalc.GetInterval();
    if (bcInterval == 0) return false;
    limit = bcInterval->UpperLimit();
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit r < " << limit << "; r max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) return false;
      r->setMax(r->getMax()*2); 
      continue;
    }
    std::cout << "\n -- Bayesian, flat prior -- " << "\n";
    std::cout << "Limit: r < " << limit << " @ 95% CL" << std::endl;
    if (0 && verbose_) {
      TCanvas c1("c1", "c1");
      RooPlot *bcPlot = bcalc.GetPosteriorPlot(true,0.1); 
      bcPlot->Draw(); 
      c1.Print("plots/bc_plot.png");
    }
    break;
  }
  return true;
}
