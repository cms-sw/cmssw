#include <stdexcept>
#include <cmath>
#include "../interface/BayesianToyMC.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooUniform.h"
#include "RooWorkspace.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"
#include "RooStats/ModelConfig.h"
#include "../interface/Combine.h"
#include "../interface/utils.h"

using namespace RooStats;

int BayesianToyMC::numIters_ = 1000;
std::string BayesianToyMC::integrationType_ = "toymc";
unsigned int BayesianToyMC::tries_ = 1;
float BayesianToyMC::hintSafetyFactor_ = 5.;

BayesianToyMC::BayesianToyMC() :
    LimitAlgo("BayesianToyMC specific options")
{
    options_.add_options()
        ("integrationType", boost::program_options::value<std::string>(&integrationType_)->default_value(integrationType_), "Integration algorithm to use")
        ("tries",      boost::program_options::value<unsigned int>(&tries_)->default_value(tries_), "Number of times to run the ToyMC on the same data")
        ("numIters,i", boost::program_options::value<int>(&numIters_)->default_value(numIters_),    "Number of iterations or calls used within iteration (0=ROOT Default)")
        ("hintSafetyFactor",
                boost::program_options::value<float>(&hintSafetyFactor_)->default_value(hintSafetyFactor_),
                "set range of integration equal to this number of times the hinted limit")
        ;
}

void BayesianToyMC::applyOptions(const boost::program_options::variables_map &vm) {
    if (!withSystematics) {
        std::cout << "BayesianToyMC: when running with no systematics, BayesianToyMC is identical to BayesianSimple." << std::endl;
        tries_    = 1;
        numIters_ = 0;
        integrationType_ = "";
    }
}
bool BayesianToyMC::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooArgSet  poi(*mc_s->GetParametersOfInterest());

  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(hintSafetyFactor_*(*hint));
  }
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
  std::auto_ptr<RooAbsPdf> nuisancePdf;
  for (;;) {
    limit = 0; limitErr = 0; bool rerun = false;
    for (unsigned int i = 0; i < tries_; ++i) {
        BayesianCalculator bcalc(data, *mc_s);
        bcalc.SetLeftSideTailFraction(0);
        bcalc.SetConfidenceLevel(cl); 
        if (!integrationType_.empty()) bcalc.SetIntegrationType(integrationType_.c_str());
        if (numIters_) bcalc.SetNumIters(numIters_);
        if (integrationType_ == "toymc") {
            if (nuisancePdf.get() == 0)  nuisancePdf.reset(utils::makeNuisancePdf(*mc_s));
            bcalc.ForceNuisancePdf(*nuisancePdf);
        }
        // get the interval
        std::auto_ptr<SimpleInterval> bcInterval(bcalc.GetInterval());
        if (bcInterval.get() == 0) return false;
        double lim = bcInterval->UpperLimit();
        // check against bound
        if (lim >= 0.5*r->getMax()) { 
            std::cout << "Limit " << r->GetName() << " < " << lim << "; " << r->GetName() << " max < " << r->getMax() << std::endl;
            if (r->getMax()/rMax > 20) return false;
            r->setMax(r->getMax()*2); 
            rerun = true; break;
        }
        // add to running sum(x) and sum(x2)
        limit    += lim;
        limitErr += lim*lim;
        if (tries_ > 1 && verbose > 1) std::cout << " - limit from try " << i << ": " << lim << std::endl;
    }
    if (rerun) continue;
    limit /= tries_; 
    limitErr = (tries_ > 1 ? std::sqrt((limitErr/tries_ - limit*limit)/((tries_-1)*tries_)) : 0);
    if (verbose > -1) {
        std::cout << "\n -- BayesianToyMC -- " << "\n";
        if (limitErr > 0) {
            std::cout << "Limit: " << r->GetName() << " < " << limit << " +/- " << limitErr << " @ " << cl * 100 << "% CL" << std::endl;
        } else {
            std::cout << "Limit: " << r->GetName() << " < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
        }
    }
    break;
  }
  return true;
}
