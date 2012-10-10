#include <stdexcept>

#include "../interface/Asymptotic.h"
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooAbsPdf.h>
#include <RooFitResult.h>
#include <RooRandom.h>
#include <RooStats/ModelConfig.h>
#include <Math/DistFuncMathCore.h>
#include "../interface/Combine.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooFitGlobalKillSentry.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/ToyMCSamplerOpt.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CascadeMinimizer.h"
#include "../interface/utils.h"
#include "../interface/AsimovUtils.h"

using namespace RooStats;

double Asymptotic::rAbsAccuracy_ = 0.0005;
double Asymptotic::rRelAccuracy_ = 0.005;
std::string Asymptotic::what_ = "both"; 
bool  Asymptotic::qtilde_ = true; 
bool  Asymptotic::picky_ = false; 
bool  Asymptotic::noFitAsimov_ = false; 
bool  Asymptotic::newExpected_ = false; 
std::string Asymptotic::minosAlgo_ = "stepping"; 
std::string Asymptotic::minimizerAlgo_ = "Minuit2";
float       Asymptotic::minimizerTolerance_ = 0.01;
int         Asymptotic::minimizerStrategy_  = 0;
double Asymptotic::rValue_ = 1.0;


Asymptotic::Asymptotic() : 
LimitAlgo("Asymptotic specific options") {
    options_.add_options()
        ("rAbsAcc", boost::program_options::value<double>(&rAbsAccuracy_)->default_value(rAbsAccuracy_), "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>(&rRelAccuracy_)->default_value(rRelAccuracy_), "Relative accuracy on r to reach to terminate the scan")
        ("run", boost::program_options::value<std::string>(&what_)->default_value(what_), "What to run: both (default), observed, expected, blind.")
        ("singlePoint",  boost::program_options::value<double>(&rValue_),  "Just compute CLs for the given value of r")
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer used for profiling (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer used for profiling")
        ("minimizerStrategy",  boost::program_options::value<int>(&minimizerStrategy_)->default_value(minimizerStrategy_),      "Stragegy for minimizer")
        ("qtilde", boost::program_options::value<bool>(&qtilde_)->default_value(qtilde_),  "Allow only non-negative signal strengths (default is true).")
        ("picky", "Abort on fit failures")
        ("noFitAsimov", "Use the pre-fit asimov dataset")
        ("newExpected", "Use the new formula for expected limits")
        ("minosAlgo", boost::program_options::value<std::string>(&minosAlgo_)->default_value(minosAlgo_), "Algorithm to use to get the median expected limit: 'minos' (fastest), 'bisection', 'stepping' (default, most robust)")
    ;
}

void Asymptotic::applyOptions(const boost::program_options::variables_map &vm) {
    if (vm.count("singlePoint") && !vm["singlePoint"].defaulted()) {
        if (!vm["run"].defaulted()) throw std::invalid_argument("Asymptotic: when using --singlePoint you can't use --run (at least for now)");
        what_ = "singlePoint";
    } else {
        if (what_ != "observed" && what_ != "expected" && what_ != "both" && what_ != "blind") 
            throw std::invalid_argument("Asymptotic: option 'run' can only be 'observed', 'expected' or 'both' (the default) or 'blind' (a-priori expected)");
    }
    picky_ = vm.count("picky");
    noFitAsimov_ = vm.count("noFitAsimov");
    newExpected_ = vm.count("newExpected");
    if (what_ == "blind") { what_ = "expected"; noFitAsimov_ = true; } 
    if (noFitAsimov_) std::cout << "Will use a-priori expected background instead of a-posteriori one." << std::endl; 
}

void Asymptotic::applyDefaultOptions() { 
    what_ = "observed"; noFitAsimov_ = true; // faster
}

bool Asymptotic::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooFitGlobalKillSentry silence(verbose <= 1 ? RooFit::WARNING : RooFit::DEBUG);
    ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
    if (verbose > 0) std::cout << "Will compute " << what_ << " limit(s) using minimizer " << minimizerAlgo_ 
                        << " with strategy " << minimizerStrategy_ << " and tolerance " << minimizerTolerance_ << std::endl;
    
    bool ret = false; 
    std::vector<std::pair<float,float> > expected;
    if (what_ == "both" || what_ == "expected") expected = runLimitExpected(w, mc_s, mc_b, data, limit, limitErr, hint);
    if (what_ != "expected") ret = runLimit(w, mc_s, mc_b, data, limit, limitErr, hint);

    if (verbose >= 0) {
        const char *rname = mc_s->GetParametersOfInterest()->first()->GetName();
        std::cout << "\n -- Asymptotic -- " << "\n";
        if (what_ == "singlePoint") {
            printf("Observed CLs for %s = %.1f: %6.4f \n", rname, rValue_, limit);
        } else if (ret && what_ != "expected") {
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

bool Asymptotic::runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first()); 
  w->loadSnapshot("clean");
  RooAbsData &asimov = *asimovDataset(w, mc_s, mc_b, data);

  r->setConstant(false);
  r->setVal(0.1*r->getMax());
  r->setMin(qtilde_ ? 0 : -r->getMax());
 
  if (params_.get() == 0) params_.reset(mc_s->GetPdf()->getParameters(data));

  hasFloatParams_ = false;
  std::auto_ptr<TIterator> itparam(params_->createIterator());
  for (RooAbsArg *a = (RooAbsArg *) itparam->Next(); a != 0; a = (RooAbsArg *) itparam->Next()) {
      RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
      if ( rrv != 0 && rrv != r && rrv->isConstant() == false ) { hasFloatParams_ = true; break; }
  }
    

  RooArgSet constraints; if (withSystematics) constraints.add(*mc_s->GetNuisanceParameters());
  nllD_.reset(mc_s->GetPdf()->createNLL(data,   RooFit::Constrain(constraints)));
  nllA_.reset(mc_s->GetPdf()->createNLL(asimov, RooFit::Constrain(constraints)));

  if (verbose > 0) std::cout << (qtilde_ ? "Restricting" : "Not restricting") << " " << r->GetName() << " to positive values." << std::endl;
  if (verbose > 1) params_->Print("V");
 
  if (verbose > 0) std::cout << "\nMake global fit of real data" << std::endl;
  {
    CloseCoutSentry sentry(verbose < 3);
    *params_ = snapGlobalObsData;
    CascadeMinimizer minim(*nllD_, CascadeMinimizer::Unconstrained, r);
    minim.setStrategy(minimizerStrategy_);
    minim.minimize(verbose-2);
    fitFreeD_.readFrom(*params_);
    minNllD_ = nllD_->getVal();
  }
  if (verbose > 0) std::cout << "NLL at global minimum of data: " << minNllD_ << " (" << r->GetName() << " = " << r->getVal() << ")" << std::endl;
  double rErr = std::max<double>(r->getError(), 0.02 * (r->getMax() - r->getMin()));

  r->setMin(0);

  if (verbose > 1) fitFreeD_.Print("V");
  if (verbose > 0) std::cout << "\nMake global fit of asimov data" << std::endl;
  {
    CloseCoutSentry sentry(verbose < 3);
    *params_ = snapGlobalObsAsimov;
    CascadeMinimizer minim(*nllA_, CascadeMinimizer::Unconstrained, r);
    minim.setStrategy(minimizerStrategy_);
    minim.minimize(verbose-2);
    fitFreeA_.readFrom(*params_);
    minNllA_ = nllA_->getVal();
  }
  if (verbose > 0) std::cout << "NLL at global minimum of asimov: " << minNllA_ << " (" << r->GetName() << " = " << r->getVal() << ")" << std::endl;
  if (verbose > 1) fitFreeA_.Print("V");

  fitFreeD_.writeTo(*params_);
  r->setConstant(true);

  if (what_ == "singlePoint") {
    limit = getCLs(*r, rValue_, true, &limit, &limitErr);
    return true;
  }

  double clsTarget = 1-cl;
  double rMin = std::max<double>(0, r->getVal()), rMax = rMin + 3 * rErr;
  double clsMax = 1, clsMin = 0;
  for (int tries = 0; tries < 5; ++tries) {
    double cls = getCLs(*r, rMax);
    if (cls == -999) { std::cerr << "Minimization failed in an unrecoverable way" << std::endl; break; }
    if (cls < clsTarget) { clsMin = cls; break; }
    rMax *= 2;
  }
  
  do {
    if (clsMax < 3*clsTarget && clsMin > 0.3*clsTarget) {
        double rCross = rMin + (rMax-rMin)*log(clsMax/clsTarget)/log(clsMax/clsMin);
        if ((rCross-rMin) < (rMax - rCross)) {
            limit = 0.8*rCross + 0.2*rMax;
        } else {
            limit = 0.8*rCross + 0.2*rMin;
        }
        limitErr = 0.5*(rMax - rMin);
    } else {
        limit = 0.5*(rMin + rMax); 
        limitErr = 0.5*(rMax - rMin);
    }
    double cls = getCLs(*r, limit);
    if (cls == -999) { std::cerr << "Minimization failed in an unrecoverable way" << std::endl; break; }
    if (cls > clsTarget) {
        clsMax = cls;
        rMin = limit;
    } else {
        clsMin = cls;
        rMax = limit;
    }
  } while (limitErr > std::max(rRelAccuracy_ * limit, rAbsAccuracy_));

  return true;
}

double Asymptotic::getCLs(RooRealVar &r, double rVal, bool getAlsoExpected, double *limit, double *limitErr) {
  r.setMax(1.1 * rVal);
  r.setConstant(true);

  CloseCoutSentry sentry(verbose < 3);

  CascadeMinimizer minimD(*nllD_, CascadeMinimizer::Constrained, &r);
  minimD.setStrategy(minimizerStrategy_);  

  (!fitFixD_.empty() ? fitFixD_ : fitFreeD_).writeTo(*params_);
  *params_ = snapGlobalObsData;
  r.setVal(rVal);
  r.setConstant(true);
  if (hasFloatParams_) {
      if (!minimD.improve(verbose-2) && picky_) return -999;
      fitFixD_.readFrom(*params_);
      if (verbose >= 2) fitFixD_.Print("V");
  }
  double qmu = 2*(nllD_->getVal() - minNllD_); if (qmu < 0) qmu = 0;

  CascadeMinimizer minimA(*nllA_, CascadeMinimizer::Constrained, &r);
  minimA.setStrategy(minimizerStrategy_); 

  (!fitFixA_.empty() ? fitFixA_ : fitFreeA_).writeTo(*params_);
  *params_ = snapGlobalObsAsimov;
  r.setVal(rVal);
  r.setConstant(true);
  if (hasFloatParams_) {
      if (!minimA.improve(verbose-2) && picky_) return -999;
      fitFixA_.readFrom(*params_);
      if (verbose >= 2) fitFixA_.Print("V");
  }
  double qA  = 2*(nllA_->getVal() - minNllA_); if (qA < 0) qA = 0;

  double CLsb = ROOT::Math::normal_cdf_c(sqrt(qmu));
  double CLb  = ROOT::Math::normal_cdf(sqrt(qA)-sqrt(qmu));
  if (qtilde_ && qmu > qA) {
    // In this region, things are tricky
    double mos = sqrt(qA); // mu/sigma
    CLsb = ROOT::Math::normal_cdf_c( (qmu + qA)/(2*mos) );
    CLb  = ROOT::Math::normal_cdf_c( (qmu - qA)/(2*mos) );
  }
  double CLs  = (CLb == 0 ? 0 : CLsb/CLb);
  sentry.clear();
  if (verbose > 0) printf("At %s = %f:\tq_mu = %.5f\tq_A  = %.5f\tCLsb = %7.5f\tCLb  = %7.5f\tCLs  = %7.5f\n", r.GetName(), rVal, qmu, qA, CLsb, CLb, CLs);

  if (getAlsoExpected) {
    const double quantiles[5] = { 0.025, 0.16, 0.50, 0.84, 0.975 };
    for (int iq = 0; iq < 5; ++iq) {
        double N = ROOT::Math::normal_quantile(quantiles[iq], 1.0);
        double clb = quantiles[iq];
        double clsplusb = ROOT::Math::normal_cdf_c( sqrt(qA) - N, 1.);
        *limit = (clb != 0 ? clsplusb/clb : 0); *limitErr = 0;
        Combine::commitPoint(true, quantiles[iq]);
        if (verbose > 0) printf("Expected %4.1f%%: CLsb = %.5f  CLb = %.5f   CLs = %.5f\n", quantiles[iq]*100, clsplusb, clb, clsplusb/clb);
    }
  }
  return CLs; 
}   

std::vector<std::pair<float,float> > Asymptotic::runLimitExpected(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
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

    // 2b) load asimov global observables
    if (params_.get() == 0) params_.reset(mc_s->GetPdf()->getParameters(data));
    *params_ = snapGlobalObsAsimov;

    // 3) solve for q_mu
    r->setConstant(false);
    //r->setMin(0);
    r->setMin(qtilde_ ? 0 : -r->getMax()); // FIXME TEST
    r->setVal(0.01*r->getMax());
    r->setError(0.1*r->getMax());
    //r->removeMax();
    
    std::auto_ptr<RooAbsReal> nll(mc_s->GetPdf()->createNLL(*asimov, RooFit::Constrain(*mc_s->GetNuisanceParameters())));
    CascadeMinimizer minim(*nll, CascadeMinimizer::Unconstrained, r);
    minim.setStrategy(minimizerStrategy_);
    minim.setErrorLevel(0.5*pow(ROOT::Math::normal_quantile(1-0.5*(1-cl),1.0), 2)); // the 0.5 is because qmu is -2*NLL
                        // eventually if cl = 0.95 this is the usual 1.92!
    CloseCoutSentry sentry(verbose < 3);    
    minim.minimize(verbose-2);
    sentry.clear();
    if (verbose > 1) {
        std::cout << "Fit to asimov dataset:" << std::endl;
        std::auto_ptr<RooFitResult> res(minim.save());
        res->Print("V");
    }
    if (r->getVal()/r->getMax() > 1e-3) {
        if (verbose) printf("WARNING: Best fit of asimov dataset is at %s = %f (%f times %sMax), while it should be at zero\n",
                r->GetName(), r->getVal(), r->getVal()/r->getMax(), r->GetName());
    }


    // 3) get ingredients for equation 37
    double nll0 = nll->getVal();
    double median = findExpectedLimitFromCrossing(*nll, r, r->getMin(), r->getMax(), nll0, 0.5);
    double sigma  = median / ROOT::Math::normal_quantile(1-0.5*(1-cl),1.0);
    double alpha = 1-cl;
    if (verbose > 0) { 
        std::cout << "Median for expected limits: " << median << std::endl; 
        std::cout << "Sigma  for expected limits: " << sigma  << std::endl; 
    }

    std::vector<std::pair<float,float> > expected;
    const double quantiles[5] = { 0.025, 0.16, 0.50, 0.84, 0.975 };
    for (int iq = 0; iq < 5; ++iq) {
        double N = ROOT::Math::normal_quantile(quantiles[iq], 1.0);
        if (newExpected_ && iq != 2) { // the median is exactly the same in the two methods
            std::string minosAlgoBackup = minosAlgo_;
            if (minosAlgo_ == "stepping") minosAlgo_ = "bisection";
            switch (iq) {
                case 0: limit = findExpectedLimitFromCrossing(*nll, r, r->getMin(),            median,      nll0, quantiles[iq]); break;
                case 1: limit = findExpectedLimitFromCrossing(*nll, r, expected.back().second, median,      nll0, quantiles[iq]); break;
                case 3: limit = findExpectedLimitFromCrossing(*nll, r, expected.back().second, median+2*sigma, nll0, quantiles[iq]); break;
                case 4: limit = findExpectedLimitFromCrossing(*nll, r, expected.back().second, median+4*sigma, nll0, quantiles[iq]); break;
            }
            minosAlgo_ = minosAlgoBackup;
            if (std::isnan(limit)) { expected.clear(); break; } 
        } else {
            limit = sigma*(ROOT::Math::normal_quantile(1 - alpha * quantiles[iq], 1.0) + N);
        }
        limitErr = 0;
        Combine::commitPoint(true, quantiles[iq]);
        expected.push_back(std::pair<float,float>(quantiles[iq], limit));
    }
    return expected;

}

float Asymptotic::findExpectedLimitFromCrossing(RooAbsReal &nll, RooRealVar *r, double rMin, double rMax, double nll0, double clb) {
    // EQ 37 of CMS NOTE 2011-005:
    //   mu_N = sigma * ( normal_quantile_c( (1-cl) * normal_cdf(N) ) + N )
    // --> (mu_N/sigma) = N + normal_quantile_c( (1-cl) * clb )
    // but qmu = (mu_N/sigma)^2
    // --> qmu = [ N + normal_quantile_c( (1-cl)*CLb ) ]^2
    // remember that qmu = 2*nll
    double N = ROOT::Math::normal_quantile(clb, 1.0);
    double errorlevel = 0.5 * pow(N+ROOT::Math::normal_quantile_c(clb*(1-cl),1.0), 2);
    int minosStat = -1;
    if (minosAlgo_ == "minos") {
        double rMax0 = r->getMax();
        // Have to repeat the fit, but I'm already at the minimum
        CascadeMinimizer minim(nll, CascadeMinimizer::Unconstrained, r);
        minim.setStrategy(minimizerStrategy_);
        minim.setErrorLevel(errorlevel); 
        CloseCoutSentry sentry(verbose < 3);
        minim.minimize(verbose-2);
        sentry.clear();
        for (int tries = 0; tries < 3; ++tries) {
            minosStat = minim.minimizer().minos(RooArgSet(*r));
            if (minosStat != -1) {
                while ((minosStat != -1) && (r->getVal()+r->getAsymErrorHi())/r->getMax() > 0.9) {
                    if (r->getMax() >= 100*rMax0) { minosStat = -1; break; }
                    r->setMax(2*r->getMax());
                    CascadeMinimizer minim2(nll, CascadeMinimizer::Unconstrained, r);
                    minim2.setStrategy(minimizerStrategy_);
                    minim2.setErrorLevel(errorlevel); 
                    minim2.minimize(verbose-2);
                    minosStat = minim2.minimizer().minos(RooArgSet(*r));
                }
                break;
            }
            minim.setStrategy(2);
            if (tries == 1) { 
                if (minimizerAlgo_.find("Minuit2") != std::string::npos) {
                    minim.minimizer().minimize("Minuit","minimize");
                } else {
                    minim.minimizer().minimize("Minuit2","minmize");
                }
            }
        }
        if (minosStat != -1) return r->getVal()+r->getAsymErrorHi();
    } else {
        double threshold = nll0 + errorlevel;
        double rCross = 0.5*(rMin+rMax), rErr = 0.5*(rMax-rMin);
        r->setVal(rCross); r->setConstant(true);
        CascadeMinimizer minim2(nll, CascadeMinimizer::Constrained);
        minim2.setStrategy(minimizerStrategy_);
        if (minosAlgo_ == "bisection") {
            if (verbose > 1) printf("Will search for NLL crossing by bisection\n");
            while (rErr > std::max(rRelAccuracy_*rCross, rAbsAccuracy_)) {
                if (rCross >= r->getMax()) r->setMax(rCross*1.1);
                r->setVal(rCross);
                bool ok = true;
                { 
                    CloseCoutSentry sentry2(verbose < 3);
                    ok = minim2.improve(verbose-2);
                }
                if (!ok && picky_) break; else minosStat = 0;
                double here = nll.getVal();
                if (verbose > 1) printf("At %s = %f:\tdelta(nll) = %.5f\n", r->GetName(), rCross, here-nll0);
                if (fabs(here - threshold) < 0.05*minimizerTolerance_) break;
                if (here < threshold) rMin = rCross; else rMax = rCross;
                rCross = 0.5*(rMin+rMax); rErr = 0.5*(rMax-rMin);
            } 
        } else if (minosAlgo_ == "stepping") {
            if (verbose > 1) printf("Will search for NLL crossing by stepping.\n");
            rCross = 0.05 * rMax; rErr = rMax; 
            double stride = rCross; bool overstepped = false;
            while (rErr > std::max(rRelAccuracy_*rCross, rAbsAccuracy_)) {
                if (rCross >= r->getMax()) r->setMax(rCross*1.1);
                double there = nll.getVal();
                r->setVal(rCross);
                bool ok = true;
                { 
                    CloseCoutSentry sentry2(verbose < 3);
                    ok = minim2.improve(verbose-2);
                }
                if (!ok && picky_) break; else minosStat = 0;
                double here = nll.getVal();
                if (verbose > 1) printf("At %s = %f:\tdelta(nll) = %.5f\n", r->GetName(), rCross, here-nll0);
                if (fabs(here - threshold) < 0.05*minimizerTolerance_) break;
                if (here < threshold) { 
                    if ((threshold-here) < 0.5*fabs(threshold-there)) stride *= 0.5;
                    rCross += stride; 
                } else { 
                    stride *= 0.5; overstepped = true;
                    rCross -= stride;
                }
                if (overstepped) rErr = stride;
            }
        } else if (minosAlgo_ == "new") {
            if (verbose > 1) printf("Will search for NLL crossing with new algorithm.\n");
            // 
            // Let X(x,y) = (x-a*y)^2 / s^2 + y^2    be the chi-square in case of correlations
            // then yhat(x) = a*x / (a^2 + s^2)
            // and X(x, yhat(x)) = x^2 / (a^2 + s^2)
            // For an unprofiled step
            //    X(x+dx, yhat(x)) - X(x,yhat(x))     = dx^2 / s^2         + 2 * x * dx / (a^2 + s^2)
            // For a profiled step  
            //    X(x+dx, yhat(x+dx)) - X(x,yhat(x))  = dx^2 / (a^2 + s^2) + 2 * x * dx / (a^2 + s^2)
            // So,
            //     dX_prof - dX_unprof = dx^2 * a^2 / (s^2 * (a^2 + s^2) )
            // The idea is then to take this approximation
            //     X_approx(x)  =  X(x, y(x1)) - k * (x-x1)^2 
            //           with k = [ X(x1, y1(x0)) - X(y1, y1(x1) ] / (x1-x0)^2
            double r_0   = rMin; 
            r->setVal(rMin);
            double nll_0 = nll.getVal(); 
            double rMax0 = rMax*100;
            double kappa = 0;
            // part 1: try to bracket the crossing between two points that have profiled nll above & below threshold
            double rStep = 0.05 * (rMax - r_0);
            double r_1 = r_0, nll_1 = nll_0;
            do {
                r_1 += rStep; 
                if (r_1 >= r->getMax()) r->setMax(r_1*1.1);
                r->setVal(r_1);
                nll_1 = nll.getVal();
                // we profile if the NLL changed by more than 0.5, or if we got above threshold
                bool binNLLchange = (nll_1 < threshold && nll_1 - nll_0 > 0.5);
                bool aboveThresh  = (nll_1 > threshold + kappa*std::pow(r_1-r_0,2));
                if (binNLLchange || aboveThresh) { 
                    if (verbose > 1) printf("At %s = %f:\tdelta(nll unprof) = %.5f\t                         \tkappa=%.5f\n", r->GetName(), r_1, nll_1-nll0, kappa);
                    { 
                        CloseCoutSentry sentry2(verbose < 3);
                        bool ok = minim2.improve(verbose-2);
                        if (!ok && picky_) return std::numeric_limits<float>::quiet_NaN();
                    }
                    double nll_1_prof = nll.getVal();
                    kappa = (nll_1 - nll_1_prof) / std::pow(r_1 - r_0,2);
                    if (verbose > 1) printf("At %s = %f:\tdelta(nll unprof) = %.5f\tdelta(nll prof) = %.5f\tkappa=%.5f\n", r->GetName(), r_1, nll_1-nll0, nll.getVal()-nll0, kappa);
                    if (nll_1_prof > threshold) { 
                        nll_1 = nll_1_prof; 
                        break; 
                    } else {
                        r_0 = r_1; 
                        nll_0 = nll_1_prof;
                        if (aboveThresh) rStep *= 2;
                    }
                } else {
                    if (verbose > 1) printf("At %s = %f:\tdelta(nll unprof) = %.5f\t                         \tkappa=%.5f\n", r->GetName(), r_1, nll_1-nll0, kappa);
                }
                if (r_1 > rMax0) return std::numeric_limits<float>::quiet_NaN();
            } while (true);
            // now crossing is bracketed, do bisection
            if (verbose > 1) printf("At %s = %f:\t                         \tdelta(nll prof) = %.5f\tkappa=%.5f\n", r->GetName(), r_0, nll_0-nll0, kappa);
            if (verbose > 1) printf("At %s = %f:\t                         \tdelta(nll prof) = %.5f\tkappa=%.5f\n", r->GetName(), r_1, nll_1-nll0, kappa);
            minosStat = 0;
            do {
               // LOOP PRECONDITIONS:
               //   - r_0 and r_1 have profiled nll values on the two sides of the threshold
               //   - nuisance parameters have been profiled at r_1 
               double rEps = 0.2*std::max(rRelAccuracy_*r_1, rAbsAccuracy_);
               // bisection loop to find point with the right nll_approx
               double r_lo = std::min(r_0,r_1), r_hi = std::max(r_1,r_0);
               while (r_hi - r_lo > rEps) {
                   double r_2 = 0.5*(r_hi+r_lo); 
                   r->setVal(r_2);
                   double y0 = nll.getVal(), y = y0 - kappa*std::pow(r_2-r_1,2);
                   if (verbose > 1) printf("At %s = %f:\tdelta(nll unprof) = %.5f\tdelta(nll appr) = %.5f\tkappa=%.5f\n", r->GetName(), r_2, y0-nll0, y-nll0, kappa);
                   if (y < threshold) { r_lo = r_2; } else { r_hi = r_2; }
               } 
               // profile at that point
               rCross = r->getVal(); 
               double nll_unprof = nll.getVal();
               bool ok = true;
               { 
                   CloseCoutSentry sentry2(verbose < 3);
                   ok = minim2.improve(verbose-2);
               }
               if (!ok && picky_) return std::numeric_limits<float>::quiet_NaN();
               double nll_prof = nll.getVal();
               if (verbose > 1) printf("At %s = %f:\tdelta(nll unprof) = %.5f\tdelta(nll prof) = %.5f\tdelta(nll appr) = %.5f\n", r->GetName(), rCross, nll_unprof-nll0, nll_prof-nll0, nll_unprof-nll0 - kappa*std::pow(rCross-r_1,2));
               if (fabs(nll_prof - threshold) < 0.1*minimizerTolerance_) { break; }
               // not yet bang on, so update r_0, kappa
               kappa = (nll_unprof - nll_prof)/std::pow(rCross-r_1,2);
               // (r0 or r1) --> r0, and rCross --> r1;  
               if ((nll_prof < threshold) == (nll_0 < threshold)) { // if rCross is on the same side of r_0
                   r_0 = r_1;   
                   nll_0 = nll_1; 
               } else {
                   // stay with r_0 as is
               }
               r_1 = rCross; nll_1 = nll_prof;
            } while (fabs(r_1-r_0) > std::max(rRelAccuracy_*rCross, rAbsAccuracy_));
        }
        if (minosStat != -1) return rCross;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

RooAbsData * Asymptotic::asimovDataset(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data) {
    // Do this only once
    if (w->data("_Asymptotic_asimovDataset_") != 0) {
        return w->data("_Asymptotic_asimovDataset_");
    }
    // snapshot data global observables
    RooArgSet gobs;
    if (withSystematics && mc_s->GetGlobalObservables()) {
        gobs.add(*mc_s->GetGlobalObservables());
        snapGlobalObsData.removeAll();
        utils::setAllConstant(gobs, true);
        gobs.snapshot(snapGlobalObsData);
    }
    // get asimov dataset and global observables
    RooAbsData *asimovData = (noFitAsimov_  ? asimovutils::asimovDatasetNominal(mc_s, 0.0, verbose) :
                                              asimovutils::asimovDatasetWithFit(mc_s, data, snapGlobalObsAsimov, 0.0, verbose));
    asimovData->SetName("_Asymptotic_asimovDataset_");
    w->import(*asimovData); // I'm assuming the Workspace takes ownership. Might be false.
    delete asimovData;      //  ^^^^^^^^----- now assuming that the workspace clones.
    return w->data("_Asymptotic_asimovDataset_");
}
