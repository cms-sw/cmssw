#include <stdexcept>
#include "../interface/BayesianFlatPrior.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooUniform.h"
#include "RooWorkspace.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "../interface/Combine.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"
#include "RooStats/ModelConfig.h"

#include <stdexcept>
#include <iostream>

using namespace RooStats;

int BayesianFlatPrior::maxDim_ = 4;
int BayesianFlatPrior::numIters_ = 0;
std::string BayesianFlatPrior::interationType_ = "default";

BayesianFlatPrior::BayesianFlatPrior() :
    LimitAlgo("BayesianSimple specific options")
{
    options_.add_options()
        ("maxDim", boost::program_options::value<int>(&maxDim_)->default_value(maxDim_), "Maximum number of dimensions to try doing the integration")
        ("interationType", boost::program_options::value<std::string>(&interationType_)->default_value(interationType_), "Integration algorithm to use")
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,29,0)
        ("numIters", boost::program_options::value<int>(&numIters_)->default_value(numIters_), "Number of iterations or calls used within iteration (0=ROOT Default)")
#endif
        ;
}

bool BayesianFlatPrior::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooArgSet  poi(*mc_s->GetParametersOfInterest());

  int dim = poi.getSize();
  if (withSystematics) dim += mc_s->GetNuisanceParameters()->getSize();
  if (dim >= maxDim_) {
    std::cerr << "ERROR: Your model has more parameters than the maximum allowed in the BayesianSimple method. \n" << 
                 "             N(params) = " << dim << ", maxDim = " << maxDim_ << "\n" <<
                 "       Please use MarkovChainMC method to compute Bayesian limits instead of BayesianSimple. \n" <<
                 "       If you really want to run BayesianSimple, change the value of the 'maxDim' option, \n" 
                 "       but note that it's really not supposed to work for N(params) above 5 or so " << std::endl;
    throw std::logic_error("Too many parameters for BayesianSimple method. Use MarkovChainMC method to compute Bayesian limits instead.");
  }

  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
  double rMax = r->getMax();
  std::auto_ptr<RooStats::ModelConfig> mc_noNuis(0);
  if (!withSystematics && mc_s->GetNuisanceParameters() != 0) {
    mc_noNuis.reset(new RooStats::ModelConfig(w));
    mc_noNuis->SetPdf(*mc_s->GetPdf());
    mc_noNuis->SetObservables(*mc_s->GetObservables());
    mc_noNuis->SetParametersOfInterest(*mc_s->GetParametersOfInterest());
    mc_noNuis->SetPriorPdf(*mc_s->GetPriorPdf());
    mc_s = mc_noNuis.get();
  }
  for (;;) {
    BayesianCalculator bcalc(data, *mc_s);
    bcalc.SetLeftSideTailFraction(0);
    bcalc.SetConfidenceLevel(cl); 
    if (interationType_ != "default") bcalc.SetIntegrationType(interationType_.c_str());
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,29,0)
    if (numIters_) bcalc.SetNumIters(numIters_);
#endif
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
    if (verbose > 200) {
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
