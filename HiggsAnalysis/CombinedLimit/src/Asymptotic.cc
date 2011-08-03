#include <stdexcept>

#include "../interface/Asymptotic.h"
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooAbsPdf.h>
#include <RooFitResult.h>
#include <RooRandom.h>
#include <RooMinimizer.h>
#include <RooStats/ModelConfig.h>
#include <Math/DistFuncMathCore.h>
#include "../interface/Combine.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooFitGlobalKillSentry.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/ToyMCSamplerOpt.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/utils.h"

using namespace RooStats;

double Asymptotic::rAbsAccuracy_ = 0.0005;
double Asymptotic::rRelAccuracy_ = 0.005;
bool  Asymptotic::expected_ = false; 
float Asymptotic::quantileForExpected_ = 0.5; 
std::string Asymptotic::minimizerAlgo_ = "Minuit2";
float       Asymptotic::minimizerTolerance_ = 1e-2;

Asymptotic::Asymptotic() : 
LimitAlgo("Asymptotic specific options") {
    options_.add_options()
        ("rAbsAcc", boost::program_options::value<double>(&rAbsAccuracy_)->default_value(rAbsAccuracy_), "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>(&rRelAccuracy_)->default_value(rRelAccuracy_), "Relative accuracy on r to reach to terminate the scan")
        ("expected", boost::program_options::value<float>(&quantileForExpected_)->default_value(0.5), "Compute the expected limit for this quantile")
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer used for profiling (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer used for profiling")
    ;
}

void Asymptotic::applyOptions(const boost::program_options::variables_map &vm) {
    if (vm.count("expected") && !vm["expected"].defaulted()) {
        if (quantileForExpected_ <= 0 || quantileForExpected_ >= 1.0) throw std::invalid_argument("Asymptotic: the quantile for the expected limit must be between 0 and 1");
        expected_ = true;
    } else {
        expected_ = false;
    }
}

void Asymptotic::applyDefaultOptions() { 
    expected_ = false;
}

bool Asymptotic::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooFitGlobalKillSentry silence(verbose <= 1 ? RooFit::WARNING : RooFit::DEBUG);
    ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
    if (expected_) return runLimitExpected(w, mc_s, mc_b, data, limit, limitErr, hint);
    return runLimit(w, mc_s, mc_b, data, limit, limitErr, hint);
}

bool Asymptotic::runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first()); 
  w->loadSnapshot("clean");
  RooAbsData &asimov = *asimovDataset(w, mc_s, mc_b, data);

  r->setConstant(false);
 
  params_.reset(mc_s->GetPdf()->getParameters(data));
  RooArgSet constraints; if (withSystematics) constraints.add(*mc_s->GetNuisanceParameters());
  nllD_.reset(mc_s->GetPdf()->createNLL(data,   RooFit::Constrain(constraints)));
  nllA_.reset(mc_s->GetPdf()->createNLL(asimov, RooFit::Constrain(constraints)));
 
  if (verbose > 1) params_->Print("V");
 
  if (verbose > 0) std::cout << "\nMake global fit of real data" << std::endl;
  {
    CloseCoutSentry sentry(verbose < 3);
    RooMinimizer minim(*nllD_);
    minim.setStrategy(0);
    minim.setPrintLevel(-1);
    nllutils::robustMinimize(*nllD_, minim, verbose-1);
    fitFreeD_.reset(minim.save());
  }
  if (verbose > 0) std::cout << "NLL at global minimum of data: " << fitFreeD_->minNll() << " (" << r->GetName() << " = " << r->getVal() << ")" << std::endl;
  if (verbose > 1) fitFreeD_->Print("V");
  if (verbose > 0) std::cout << "\nMake global fit of asimov data" << std::endl;
  {
    CloseCoutSentry sentry(verbose < 3);
    RooMinimizer minim(*nllA_);
    minim.setStrategy(0);
    minim.setPrintLevel(-1);
    nllutils::robustMinimize(*nllA_, minim, verbose-1);
    fitFreeA_.reset(minim.save());
  }
  if (verbose > 0) std::cout << "NLL at global minimum of asimov: " << fitFreeA_->minNll() << " (" << r->GetName() << " = " << r->getVal() << ")" << std::endl;
  if (verbose > 1) fitFreeA_->Print("V");

  *params_ = fitFreeD_->floatParsFinal();
  r->setConstant(true);

  double clsTarget = 1-cl;
  double rMin = r->getVal(), rMax = rMin + 3 * ((RooRealVar*)fitFreeD_->floatParsFinal().find(r->GetName()))->getError();
  for (int tries = 0; tries < 5; ++tries) {
    double cls = getCLs(*r, rMax);
    if (cls < clsTarget) break;
    rMax *= 2;
  }
  do {
    limit = 0.5*(rMin + rMax); limitErr = 0.5*(rMax - rMin);
    double cls = getCLs(*r, limit);
    if (cls > clsTarget) {
        rMin = limit;
    } else {
        rMax = limit;
    }
  } while (limitErr > std::max(rRelAccuracy_ * limit, rAbsAccuracy_));

  std::cout << "\n -- Asymptotic -- " << "\n";
  std::cout << "Limit: " << r->GetName() << " < " << limit << " +/- " << limitErr << " @ " << cl << "% CL" << std::endl;
  return true;
}

double Asymptotic::getCLs(RooRealVar &r, double rVal) {
  r.setMax(1.1 * rVal);
  r.setConstant(true);

  CloseCoutSentry sentry(verbose < 2);
  RooMinimizer minimD(*nllD_), minimA(*nllA_);
  minimD.setStrategy(0); minimD.setPrintLevel(-1); 
  minimA.setStrategy(0); minimA.setPrintLevel(-1);

  *params_ = fitFixD_.get() ? fitFixD_->floatParsFinal() : fitFreeD_->floatParsFinal();
  r.setVal(rVal);
  r.setConstant(true);
  nllutils::robustMinimize(*nllD_, minimD, verbose-1);
  fitFixD_.reset(minimD.save());
  if (verbose >= 2) fitFixD_->Print("V");
  double qmu = nllD_->getVal() - fitFreeD_->minNll(); if (qmu < 0) qmu = 0;

  *params_ = fitFixA_.get() ? fitFixA_->floatParsFinal() : fitFreeA_->floatParsFinal();
  r.setVal(rVal);
  r.setConstant(true);
  nllutils::robustMinimize(*nllA_, minimA, verbose-1);
  fitFixA_.reset(minimA.save());
  if (verbose >= 2) fitFixA_->Print("V");
  double qA  = nllA_->getVal() - fitFreeA_->minNll(); if (qA < 0) qA = 0;

  double CLsb = ROOT::Math::normal_cdf_c(sqrt(2*qmu));
  double CLb  = ROOT::Math::normal_cdf(sqrt(2*qA)-sqrt(2*qmu));
  double CLs  = (CLb == 0 ? 0 : CLsb/CLb);
  sentry.clear();
  if (verbose > 0) printf("At %s = %f:\n\tq_mu = %.5f\tq_A  = %.5f\n\tCLsb = %7.5f\tCLb  = %7.5f\tCLs  = %7.5f\n", r.GetName(), rVal, qmu, qA, CLsb, CLb, CLs);
  return CLs; 
}   

bool Asymptotic::runLimitExpected(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    // See equation 35-38 of AN 2011/298 and references cited therein
    //
    //   (35)  sigma^2   = mu^2 / q_mu(Asimov)
    //   (38)  mu_median = sigma * normal_quantile(1-0.5*(1-cl))
    //
    // -->  q_mu(Asimov) = pow(normal_quantile(1-0.5*(1-cl)), 2)
    //      can be solved to find mu_median
    //
    // --> then (38) gives sigma, and the quantiles are given by (37)
    //      mu_N = sigma * (normal_quantile(1 - quantile*(1-cl), 1.0) + normal_quantile(quantile));
    //
    // 1) get parameter of interest
    RooArgSet  poi(*mc_s->GetParametersOfInterest());
    RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());

    // 2) get asimov dataset
    RooAbsData *asimov = asimovDataset(w, mc_s, mc_b, data);

    // 3) solve for q_mu
    CloseCoutSentry sentry(verbose < 3);
    r->setConstant(false);
    std::auto_ptr<RooAbsReal> nll(mc_s->GetPdf()->createNLL(*asimov, RooFit::Constrain(*mc_s->GetNuisanceParameters())));
    RooMinimizer minim(*nll);
    minim.setStrategy(0);
    minim.setPrintLevel(-1);
    minim.setErrorLevel(0.5*pow(ROOT::Math::normal_quantile(1-0.5*(1-cl),1.0), 2)); // the 0.5 is because qmu is -2*NLL
                        // eventually if cl = 0.95 this is the usual 1.92!
    nllutils::robustMinimize(*nll, minim, verbose-1);
    minim.minos(RooArgSet(*r));
    sentry.clear();
    
    // 3) get ingredients for equation 37
    double median = r->getAsymErrorHi();
    double sigma  = median / ROOT::Math::normal_quantile(1-0.5*(1-cl),1.0);
    double N     = ROOT::Math::normal_quantile(quantileForExpected_, 1.0);
    double alpha = 1-cl;
    limit = sigma*(ROOT::Math::normal_quantile(1 - alpha * quantileForExpected_, 1.0) + N);
    limitErr = 0;
    std::cout << "\n -- Asymptotic -- " << "\n";
    std::cout << "Asymptotic expected limit for " << quantileForExpected_ << ": " << r->GetName() << " < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
    return true;
}

RooAbsData * Asymptotic::asimovDataset(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data) {
    if (w->data("_Asymptotic_asimovDataset_") == 0) {
        std::cout << "Generating Asimov dataset" << std::endl;
        RooArgSet  poi(*mc_s->GetParametersOfInterest());
        RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
        r->setConstant(true); r->setVal(0);
        CloseCoutSentry sentry(verbose < 3);
        mc_s->GetPdf()->fitTo(data, RooFit::Minimizer("Minuit2","minimize"), RooFit::Strategy(1), RooFit::Constrain(*mc_s->GetNuisanceParameters()));
        toymcoptutils::SimPdfGenInfo newToyMC(*mc_b->GetPdf(), *mc_s->GetObservables(), false); 
        RooRealVar *weightVar = 0;
        RooAbsData *asimov = newToyMC.generateAsimov(weightVar); // as simple as that
        asimov->SetName("_Asymptotic_asimovDataset_");
        w->import(*asimov);
        delete weightVar;
    }
    return w->data("_Asymptotic_asimovDataset_");
}
