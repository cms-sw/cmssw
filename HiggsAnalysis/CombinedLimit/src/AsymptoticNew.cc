#include <stdexcept>
#include <RooStats/ModelConfig.h>
#include <RooStats/HypoTestInverter.h>
#include <RooStats/HypoTestInverterResult.h>
#include "RooStats/AsymptoticCalculator.h"
#include "../interface/AsymptoticNew.h"
#include "../interface/Combine.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooFitGlobalKillSentry.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/utils.h"

using namespace RooStats;

std::string AsymptoticNew::what_ = "both"; 
bool  AsymptoticNew::qtilde_ = true;
int AsymptoticNew::nscanpoints_ = 20; 
double AsymptoticNew::minrscan_ = 0.; 
double AsymptoticNew::maxrscan_ = 20.; 

AsymptoticNew::AsymptoticNew() :
LimitAlgo("AsymptoticNew specific options"){
    options_.add_options()
        ("run", boost::program_options::value<std::string>(&what_)->default_value(what_), "What to run: both (default), observed, expected.")
        ("nPoints", boost::program_options::value<int>(&nscanpoints_)->default_value(nscanpoints_), "Number of points in scan for CLs")
        ("qtilde", boost::program_options::value<bool>(&qtilde_)->default_value(qtilde_),  "Allow only non-negative signal strengths (default is true).")
        ("scanMin", boost::program_options::value<double>(&minrscan_)->default_value(minrscan_),  "Minimum value for scan in r.")
        ("scanMax", boost::program_options::value<double>(&maxrscan_)->default_value(maxrscan_),  "Maximum value for scan in r.")
  ;
}

void AsymptoticNew::applyOptions(const boost::program_options::variables_map &vm) {
        if (what_ != "observed" && what_ != "expected" && what_ != "both") 
            throw std::invalid_argument("Asymptotic: option 'run' can only be 'observed', 'expected' or 'both' (the default)");

}

void AsymptoticNew::applyDefaultOptions() { 
    what_ = "observed";
    nscanpoints_ = 20;
    qtilde_ = true;
    minrscan_=0.;
    maxrscan_=20.;
}

bool AsymptoticNew::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooFitGlobalKillSentry silence(verbose <= 1 ? RooFit::WARNING : RooFit::DEBUG);
    if (verbose > 0) std::cout << "Will compute " << what_ << " limit(s) " << std::endl;
  
    bool ret = false; 
    std::vector<std::pair<float,float> > expected;
    expected = runLimit(w, mc_s, mc_b, data, limit, limitErr, hint);

    if (verbose >= 0) {
        const char *rname = mc_s->GetParametersOfInterest()->first()->GetName();
        std::cout << "\n -- AsymptotiNew -- " << "\n";
        if (ret && what_ != "expected") {
            printf("Observed Limit: %s < %6.4f\n", rname, limit);
        }
        for (std::vector<std::pair<float,float> >::const_iterator it = expected.begin(), ed = expected.end(); it != ed; ++it) {
            printf("Expected %4.1f%%: %s < %6.4f\n", it->first*100, rname, it->second);
        }
        std::cout << std::endl;
    }

    // note that for expected we have to return FALSE even if we succeed because otherwise it goes into the observed limit as well
    return ret;
}

std::vector<std::pair<float,float> > AsymptoticNew::runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
 
  RooRealVar *poi = (RooRealVar*)mc_s->GetParametersOfInterest()->first();
  mc_s->SetSnapshot(*poi); 
  double oldval = poi->getVal(); 
  poi->setVal(0); mc_b->SetSnapshot(*poi); 
  poi->setVal(oldval);

  // Set up asymptotic calculator
  AsymptoticCalculator * ac = new RooStats::AsymptoticCalculator(data, *mc_b, *mc_s);
  AsymptoticCalculator::SetPrintLevel(verbose>2);
  ac->SetOneSided(true); 
  ac->SetQTilde(qtilde_);
  HypoTestInverter calc(*ac);
  calc.SetVerbose(verbose>2);
  calc.UseCLs(true);
  
  calc.SetFixedScan(nscanpoints_,minrscan_,maxrscan_); 
  RooStats::HypoTestInverterResult * r = calc.GetInterval();

  // Expected Median + bands
  // bands will be based on Z-values
  std::vector<std::pair<float,float> > expected;
  const double quantiles[5] = { 0.025, 0.16, 0.50, 0.84, 0.975 };
  const double zvals[5]     = { -2, -1, 0, 1, 2 };

  for (int iq = 0; iq < 5; ++iq) {
        limit = r->GetExpectedUpperLimit(zvals[iq]);
        limitErr = 0;
        Combine::commitPoint(true, quantiles[iq]);
        expected.push_back(std::pair<float,float>(quantiles[iq], limit));
  }

  // Observed Limit
  limit = r->UpperLimit();
  return expected;
   
}
