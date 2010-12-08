#include "HiggsAnalysis/CombinedLimit/interface/BayesianFlatPrior.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooUniform.h"
#include "RooWorkspace.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"

using namespace RooStats;



bool BayesianFlatPrior::run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
  RooRealVar *r = w->var("r");
  RooAbsPdf *prior = w->pdf("prior"); if (prior == 0) { std::cerr << "ERROR: missing prior" << std::endl; abort(); }
  RooArgSet  poi(*r);
  double rMax = r->getMax();
  for (;;) {
    BayesianCalculator bcalc(data, *w->pdf("model_s"), poi, *prior, (withSystematics ? w->set("nuisances") : 0));
    bcalc.SetLeftSideTailFraction(0);
    bcalc.SetConfidenceLevel(cl); 
    std::auto_ptr<SimpleInterval> bcInterval(bcalc.GetInterval());
    if (bcInterval.get() == 0) return false;
    limit = bcInterval->UpperLimit();
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit r < " << limit << "; r max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) return false;
      r->setMax(r->getMax()*2); 
      continue;
    }
    if (verbose > 0) {
        std::cout << "\n -- Bayesian, flat prior -- " << "\n";
        std::cout << "Limit: r < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
    }
    if (verbose > 2) {
      TCanvas c1("c1", "c1");
      std::auto_ptr<RooPlot> bcPlot(bcalc.GetPosteriorPlot(true, 0.1)); 
      bcPlot->Draw(); 
      c1.Print("plots/bc_plot.png");
    }
    break;
  }
  return true;
}
