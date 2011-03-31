#include "HiggsAnalysis/CombinedLimit/interface/ProfileLikelihood.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooRandom.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "TCanvas.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/HypoTestResult.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/CloseCoutSentry.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"


#include <Math/MinimizerOptions.h>

using namespace RooStats;

std::string ProfileLikelihood::minimizerAlgo_ = "Minuit2";
float       ProfileLikelihood::minimizerTolerance_ = 1e-3;
int         ProfileLikelihood::tries_ = 1;
int         ProfileLikelihood::maxTries_ = 1;
float       ProfileLikelihood::maxRelDeviation_ = 0.05;
float       ProfileLikelihood::maxOutlierFraction_ = 0.25;
int         ProfileLikelihood::maxOutliers_ = 3;
bool        ProfileLikelihood::preFit_ = false;
std::string ProfileLikelihood::plot_ = "";

ProfileLikelihood::ProfileLikelihood() :
    LimitAlgo("Profile Likelihood specific options")
{
    options_.add_options()
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer")
        ("tries",              boost::program_options::value<int>(&tries_)->default_value(tries_), "Compute PL limit N times, to check for numerical instabilities")
        ("maxTries",           boost::program_options::value<int>(&maxTries_)->default_value(maxTries_), "Stop trying after N attempts per point")
        ("maxRelDeviation",    boost::program_options::value<float>(&maxRelDeviation_)->default_value(maxOutlierFraction_), "Max absolute deviation of the results from the median")
        ("maxOutlierFraction", boost::program_options::value<float>(&maxOutlierFraction_)->default_value(maxOutlierFraction_), "Ignore up to this fraction of results if they're too far from the median")
        ("maxOutliers",        boost::program_options::value<int>(&maxOutliers_)->default_value(maxOutliers_),      "Stop trying after finding N outliers")
        ("plot",   boost::program_options::value<std::string>(&plot_)->default_value(plot_), "Save a plot of the negative log of the profiled likelihood into the specified file")
        ("preFit", "Attept a fit before running the ProfileLikelihood calculator")
    ;
}

void ProfileLikelihood::applyOptions(const boost::program_options::variables_map &vm) 
{
}

ProfileLikelihood::MinimizerSentry::MinimizerSentry(std::string &minimizerAlgo, double tolerance) :
    minimizerTypeBackup(ROOT::Math::MinimizerOptions::DefaultMinimizerType()),
    minimizerAlgoBackup(ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo()),
    minimizerTollBackup(ROOT::Math::MinimizerOptions::DefaultTolerance())
{
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(tolerance);
  if (minimizerAlgo.find(",") != std::string::npos) {
      size_t idx = minimizerAlgo.find(",");
      std::string type = minimizerAlgo.substr(0,idx), algo = minimizerAlgo.substr(idx+1);
      if (verbose > 1) std::cout << "Set default minimizer to " << type << ", algorithm " << algo << std::endl;
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer(type.c_str(), algo.c_str());
  } else {
      if (verbose > 1) std::cout << "Set default minimizer to " << minimizerAlgo << std::endl;
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerAlgo.c_str());
  }
}

ProfileLikelihood::MinimizerSentry::~MinimizerSentry() 
{
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(minimizerTollBackup);
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerTypeBackup.c_str(),minimizerAlgoBackup.empty() ? 0 : minimizerAlgoBackup.c_str());
}

bool ProfileLikelihood::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
  CloseCoutSentry sentry(verbose < 0);

  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
  bool success = false;
  std::vector<double> limits; double rMax = r->getMax();  
  std::auto_ptr<RooAbsPdf> nuisancePdf(0);
  for (int i = 0; i < maxTries_; ++i) {
      w->loadSnapshot("clean");
      if (i > 0) { // randomize starting point
        r->setMax(rMax*(0.5+RooRandom::uniform()));
        r->setVal((0.1+0.5*RooRandom::uniform())*r->getMax()); 
        if (withSystematics) { 
            if (nuisancePdf.get() == 0) nuisancePdf.reset(utils::makeNuisancePdf(*mc_s));
            RooArgSet set(*mc_s->GetNuisanceParameters()); 
            RooDataSet *randoms = nuisancePdf->generate(set, 1); 
            set = *randoms->get(0);
            if (verbose > 2) {
                std::cout << "Starting minimization from point " << std::endl;
                r->Print("V");
                set.Print("V");
            }
            delete randoms;
        }
      }
      if (preFit_) {
        CloseCoutSentry sentry(verbose < 2);
        RooFitResult *res = mc_s->GetPdf()->fitTo(data, RooFit::Save(1), RooFit::Minimizer("Minuit2"));
        if (res == 0 || res->covQual() != 3 || res->edm() > minimizerTolerance_) {
            if (verbose > 1) std::cout << "Fit failed (covQual " << (res ? res->covQual() : -1) << ", edm " << (res ? res->edm() : 0) << ")" << std::endl;
            continue;
        }
        if (verbose > 1) {
            res->Print("V");
            std::cout << "Covariance quality: " << res->covQual() << ", Edm = " << res->edm() << std::endl;
        }
        delete res;
      }
      bool thisTry = (doSignificance_ ?  runSignificance(w,mc_s,data,limit,limitErr) : runLimit(w,mc_s,data,limit,limitErr));
      if (!thisTry) continue;
      if (tries_ == 1) { success = true; break; }
      limits.push_back(limit); 
      int nresults = limits.size();
      if (nresults < tries_) continue;
      std::sort(limits.begin(), limits.end());
      double median = (nresults % 2 ? limits[nresults/2] : 0.5*(limits[nresults/2] + limits[nresults/2+1]));
      int noutlier = 0; double spreadIn = 0, spreadOut = 0;
      for (int j = 0; j < nresults; ++j) {
        double diff = fabs(limits[j]-median)/median;
        if (diff < maxRelDeviation_) { 
          spreadIn = max(spreadIn, diff); 
        } else {
          noutlier++;
          spreadOut = max(spreadOut, diff); 
        }
      }
      if (verbose > 0) {
          std::cout << "Numer of tries: " << i << "   Number of successes: " << nresults 
                    << ", Outliers: " << noutlier << " (frac = " << noutlier/double(nresults) << ")"
                    << ", Spread of non-outliers: " << spreadIn <<" / of outliers: " << spreadOut << std::endl;
      }
      if (noutlier <= maxOutlierFraction_*nresults) {
        if (verbose > 0) std::cout << " \\--> success! " << std::endl;
        success = true;
        limit   = median;
        break;
      } else if (noutlier > maxOutliers_) {
        if (verbose > 0) std::cout << " \\--> failure! " << std::endl;
        break;
      }
  }
  return success;
}

bool ProfileLikelihood::runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooAbsData &data, double &limit, double &limitErr) {
  RooArgSet  poi(*mc_s->GetParametersOfInterest());
  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
  double rMax = r->getMax();
  bool success = false;
  CloseCoutSentry coutSentry(verbose <= 1); // close standard output and error, so that we don't flood them with minuit messages

  while (!success) {
    ProfileLikelihoodCalculator plcB(data, *mc_s, 1.0-cl);
    std::auto_ptr<LikelihoodInterval> plInterval(plcB.GetInterval());
    if (plInterval.get() == 0) break;
    limit = plInterval->UpperLimit(*r); 
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit " << r->GetName() << " < " << limit << "; " << r->GetName() << " max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) break;
      r->setMax(r->getMax()*2); 
      continue;
    }
    if (limit == r->getMin()) {
      std::cerr << "ProfileLikelihoodCalculator failed (returned upper limit equal to the lower bound)" << std::endl;
      break;
    }
    success = true;
    if (!plot_.empty()) {
        TCanvas *c1 = new TCanvas("c1","c1");
        LikelihoodIntervalPlot plot(&*plInterval);
        plot.Draw();
        c1->Print(plot_.c_str());
        delete c1;
    }
  }
  coutSentry.clear();
  if (verbose >= 0) {
      if (success) {
        std::cout << "\n -- Profile Likelihood -- " << "\n";
        std::cout << "Limit: " << r->GetName() << " < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
      }
  }
  return success;
}

bool ProfileLikelihood::runSignificance(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooAbsData &data, double &limit, double &limitErr) {
  RooArgSet  poi(*mc_s->GetParametersOfInterest());
  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());

  ProfileLikelihoodCalculator plcS(data, *mc_s, 1.0-cl);
  RooArgSet nullParamValues; 
  r->setVal(0.0);
  nullParamValues.addClone(*r); 
  plcS.SetNullParameters(nullParamValues);

  CloseCoutSentry coutSentry(verbose <= 1); // close standard output and error, so that we don't flood them with minuit messages
  std::auto_ptr<HypoTestResult> result(plcS.GetHypoTest());
  if (result.get() == 0) return false;
  coutSentry.clear();

  limit = result->Significance();
  if (limit == 0 && signbit(limit)) {
      std::cerr << "ProfileLikelihoodCalculator failed (returned significance -0)" << std::endl;
      return false;
  }
  std::cout << "\n -- Profile Likelihood -- " << "\n";
  std::cout << "Significance: " << limit << std::endl;
  return true;
}

