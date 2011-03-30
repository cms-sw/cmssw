#include <stdexcept>
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
#include "RooStats/ModelConfig.h"

using namespace RooStats;

bool BayesianFlatPrior::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooArgSet  poi(*mc_s->GetParametersOfInterest());
  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
  double rMax = r->getMax();
  for (;;) {
    BayesianCalculator bcalc(data, *mc_s);
    bcalc.SetLeftSideTailFraction(0);
    bcalc.SetConfidenceLevel(cl); 
    std::auto_ptr<SimpleInterval> bcInterval(bcalc.GetInterval());
    if (bcInterval.get() == 0) return false;
    limit = bcInterval->UpperLimit();
    if (limit >= 0.5*r->getMax()) { 
      std::cout << "Limit " << r->GetName() << " < " << limit << "; " << r->GetName() << " max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) return false;
      r->setMax(r->getMax()*2); 
      continue;
    }
    if (verbose > -1) {
        std::cout << "\n -- BayesianSimple -- " << "\n";
        std::cout << "Limit: " << r->GetName() << " < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
    }
    if (verbose > 2) {
      // FIXME!!!!!
      TCanvas c1("c1", "c1");
      std::auto_ptr<RooPlot> bcPlot(bcalc.GetPosteriorPlot(true, 0.1)); 
      bcPlot->Draw(); 
      c1.Print("plots/bc_plot.png");
    }
    break;
  }
  return true;
}
