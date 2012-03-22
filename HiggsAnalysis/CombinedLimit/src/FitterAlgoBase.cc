#include "../interface/FitterAlgoBase.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooRandom.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooSimultaneous.h"
#include "RooAddPdf.h"
#include "RooProdPdf.h"
#include "RooConstVar.h"
#include "RooPlot.h"
#include <RooMinimizer.h>
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2.h"
#include "TFile.h"
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CascadeMinimizer.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"

#include "../interface/ProfilingTools.h"


#include <Math/MinimizerOptions.h>

using namespace RooStats;

std::string FitterAlgoBase::minimizerAlgo_ = "Minuit2";
std::string FitterAlgoBase::minimizerAlgoForMinos_ = "Minuit2,simplex";
float       FitterAlgoBase::minimizerTolerance_ = 1e-2;
float       FitterAlgoBase::minimizerToleranceForMinos_ = 1e-4;
int         FitterAlgoBase::minimizerStrategy_  = 1;
int         FitterAlgoBase::minimizerStrategyForMinos_ = 0;
float       FitterAlgoBase::preFitValue_ = 1.0;
float       FitterAlgoBase::stepSize_ = 0.1;
bool        FitterAlgoBase::robustFit_ = false;
int         FitterAlgoBase::maxFailedSteps_ = 5;
bool        FitterAlgoBase::do95_ = false;

FitterAlgoBase::FitterAlgoBase(const char *title) :
    LimitAlgo(title)
{
    options_.add_options()
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer")
        ("minimizerStrategy",  boost::program_options::value<int>(&minimizerStrategy_)->default_value(minimizerStrategy_),      "Stragegy for minimizer")
        ("preFitValue",        boost::program_options::value<float>(&preFitValue_)->default_value(preFitValue_),  "Value of signal strength pre-fit")
        ("do95",       boost::program_options::value<bool>(&do95_)->default_value(do95_),  "Compute also 2-sigma interval from delta(nll) = 1.92 instead of 0.5")
        ("robustFit",  boost::program_options::value<bool>(&robustFit_)->default_value(robustFit_),  "Search manually for 1 and 2 sigma bands instead of using Minos")
        ("maxFailedSteps",  boost::program_options::value<int>(&maxFailedSteps_)->default_value(maxFailedSteps_),  "How many failed steps to retry before giving up")
        ("stepSize",        boost::program_options::value<float>(&stepSize_)->default_value(stepSize_),  "Step size for robust fits (multiplier of the range)")
        ("minimizerAlgoForMinos",      boost::program_options::value<std::string>(&minimizerAlgoForMinos_)->default_value(minimizerAlgoForMinos_), "Choice of minimizer (Minuit vs Minuit2) for profiling in robust fits")
        ("minimizerStrategyForMinos",  boost::program_options::value<int>(&minimizerStrategyForMinos_)->default_value(minimizerStrategyForMinos_),      "Stragegy for minimizer for profiling in robust fits")
        ("minimizerToleranceForMinos",  boost::program_options::value<float>(&minimizerToleranceForMinos_)->default_value(minimizerToleranceForMinos_),      "Tolerance for minimizer for profiling in robust fits")
    ;
}

void FitterAlgoBase::applyOptionsBase(const boost::program_options::variables_map &vm) 
{
}

bool FitterAlgoBase::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
  CloseCoutSentry sentry(verbose < 0);

  return runSpecific(w, mc_s, mc_b, data, limit, limitErr, hint);
}


RooFitResult *FitterAlgoBase::doFit(RooAbsPdf &pdf, RooAbsData &data, RooRealVar &r, const RooCmdArg &constrain, bool doHesse) {
    RooArgList rs(r);
    return doFit(pdf, data, rs, constrain, doHesse);

}

RooFitResult *FitterAlgoBase::doFit(RooAbsPdf &pdf, RooAbsData &data, RooArgList &rs, const RooCmdArg &constrain, bool doHesse) {
    RooFitResult *ret = 0;
    std::auto_ptr<RooAbsReal> nll(pdf.createNLL(data, constrain, RooFit::Extended(pdf.canBeExtended())));

    CascadeMinimizer minim(*nll, CascadeMinimizer::Unconstrained, rs.getSize() ? dynamic_cast<RooRealVar*>(rs.first()) : 0);
    minim.setStrategy(minimizerStrategy_);
    CloseCoutSentry sentry(verbose < 3);    
    bool ok = minim.minimize(verbose-1);
    if (!ok) { std::cout << "Initial minimization failed. Aborting." << std::endl; return 0; }
    if (doHesse) minim.minimizer().hesse();
    sentry.clear();
    ret = minim.save();
    if (verbose > 1) { ret->Print("V");  }

    std::auto_ptr<RooArgSet> allpars(pdf.getParameters(data));

    for (int i = 0, n = rs.getSize(); i < n; ++i) {
        // if this is not the first fit, reset parameters  
        if (i) {
            RooArgSet oldparams(ret->floatParsFinal());
            *allpars = oldparams;
        }
    
        // get the parameter to scan, amd output variable in fit result
        RooRealVar &r = dynamic_cast<RooRealVar &>(*rs.at(i));
        RooRealVar &rf = dynamic_cast<RooRealVar &>(*ret->floatParsFinal().find(r.GetName()));
        double r0 = r.getVal(), rMin = r.getMin(), rMax = r.getMax();

        if (!robustFit_) {
            if (do95_) {
                minim.setErrorLevel(1.92);
                minim.improve(verbose-1);
                if (minim.minimizer().minos(RooArgSet(r)) != -1) {
                    rf.setRange("err95", r.getVal() + r.getAsymErrorLo(), r.getVal() + r.getAsymErrorHi());
                }
                minim.setErrorLevel(0.5);
                minim.improve(verbose-1);
            }
            if (minim.minimizer().minos(RooArgSet(r)) != -1) {
                rf.setRange("err68", r.getVal() + r.getAsymErrorLo(), r.getVal() + r.getAsymErrorHi());
                rf.setAsymError(r.getAsymErrorLo(), r.getAsymErrorHi());
            }
       } else {
            r.setVal(r0); r.setConstant(true);
            
            CascadeMinimizer minim2(*nll, CascadeMinimizer::Constrained);
            minim2.setStrategy(minimizerStrategyForMinos_);

            std::auto_ptr<RooArgSet> allpars(nll->getParameters((const RooArgSet *)0));

            double nll0 = nll->getVal();
            double threshold68 = nll0 + 0.5;
            double threshold95 = nll0 + 1.92;
            // search for crossings

            assert(!std::isnan(r0));
            // high error
            double hi68 = findCrossing(minim2, *nll, r, threshold68, r0,   rMax);
            double hi95 = do95_ ? findCrossing(minim2, *nll, r, threshold95, std::isnan(hi68) ? r0 : hi68, std::max(rMax, std::isnan(hi68*2-r0) ? r0 : hi68*2-r0)) : r0;
            // low error 
            *allpars = RooArgSet(ret->floatParsFinal()); r.setVal(r0); r.setConstant(true);
            double lo68 = findCrossing(minim2, *nll, r, threshold68, r0,   rMin); 
            double lo95 = do95_ ? findCrossing(minim2, *nll, r, threshold95, std::isnan(lo68) ? r0 : lo68, rMin) : r0;

            rf.setAsymError(!std::isnan(lo68) ? lo68 - r0 : 0, !std::isnan(hi68) ? hi68 - r0 : 0);
            rf.setRange("err68", !std::isnan(lo68) ? lo68 : r0, !std::isnan(hi68) ? hi68 : r0);
            if (do95_ && (!std::isnan(lo95) || !std::isnan(hi95))) {
                rf.setRange("err95", !std::isnan(lo95) ? lo95 : r0, !std::isnan(hi95) ? hi95 : r0);
            }

            r.setVal(r0); r.setConstant(false);
        }
    }

    return ret;
}

double FitterAlgoBase::findCrossing(CascadeMinimizer &minim, RooAbsReal &nll, RooRealVar &r, double level, double rStart, double rBound) {
    ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgoForMinos_, minimizerToleranceForMinos_);
    if (verbose) std::cout << "Searching for crossing at nll = " << level << " in the interval " << rStart << ", " << rBound << std::endl; 
    double rInc = stepSize_*(rBound - rStart);
    r.setVal(rStart); 
    std::auto_ptr<RooFitResult> checkpoint;
    std::auto_ptr<RooArgSet>    allpars;
    bool ok = false;
    {
        CloseCoutSentry sentry(verbose < 3);    
        ok = minim.improve(verbose-1);
        checkpoint.reset(minim.save());
    }
    if (!ok) { std::cout << "Error: minimization failed at " << r.GetName() << " = " << rStart << std::endl; return NAN; }
    double here = nll.getVal();
    int nfail = 0;
    if (verbose > 0) { printf("      %s      lvl-here  lvl-there   stepping\n", r.GetName()); fflush(stdout); }
    do {
        rStart += rInc;
        if (rInc*(rStart - rBound) > 0) { // went beyond bounds
            rStart -= rInc;
            rInc    = 0.5*(rBound-rStart);
        }
        r.setVal(rStart);
        nll.clearEvalErrorLog(); nll.getVal();
        if (nll.numEvalErrors() > 0) {
            ok = false;
        } else {
            CloseCoutSentry sentry(verbose < 3);    
            ok = minim.improve(verbose-1);
        }
        if (!ok) { 
            nfail++;
            if (nfail >= maxFailedSteps_) {  std::cout << "Error: minimization failed at " << r.GetName() << " = " << rStart << std::endl; return NAN; }
            RooArgSet oldparams(checkpoint->floatParsFinal());
            if (allpars.get() == 0) allpars.reset(nll.getParameters((const RooArgSet *)0));
            *allpars = oldparams;
            rStart -= rInc; rInc *= 0.5; 
            continue;
        } else nfail = 0;
        double there = here;
        here = nll.getVal();
        if (verbose > 0) { printf("%f    %+.5f  %+.5f    %f\n", rStart, level-here, level-there, rInc); fflush(stdout); }
        if ( fabs(here - level) < 4*minimizerToleranceForMinos_ ) {
            // set to the right point with interpolation
            r.setVal(rStart + (level-here)*(level-there)/(here-there));
            return r.getVal();
        } else if (here > level) {
            // I'm above the level that I wanted, this means I stepped too long
            // First I take back all the step
            rStart -= rInc; 
            // Then I try to figure out a better step
            if (runtimedef::get("FITTER_DYN_STEP")) {
                if (fabs(there - level) > 0.05) { // If started from far away, I still have to step carefully
                    double rIncFactor = std::max(0.2, std::min(0.7, 0.75*(level-there)/(here-there)));
                    //printf("\t\t\t\t\tCase A1: level-there = %f,  here-there = %f,   rInc(Old) = %f,  rInFactor = %f,  rInc(New) = %f\n", level-there, here-there, rInc, rIncFactor, rInc*rIncFactor);
                    rInc *= rIncFactor;
                } else { // close enough to jump straight to target
                    double rIncFactor = std::max(0.05, std::min(0.95, 0.95*(level-there)/(here-there)));
                    //printf("\t\t\t\t\tCase A2: level-there = %f,  here-there = %f,   rInc(Old) = %f,  rInFactor = %f,  rInc(New) = %f\n", level-there, here-there, rInc, rIncFactor, rInc*rIncFactor);
                    rInc *= rIncFactor;
                }
            } else {
                rInc *= 0.3;
            }
            if (allpars.get() == 0) allpars.reset(nll.getParameters((const RooArgSet *)0));
            RooArgSet oldparams(checkpoint->floatParsFinal());
            *allpars = oldparams;
        } else if ((here-there)*(level-there) < 0 && // went wrong
                   fabs(here-there) > 0.1) {         // by more than roundoff
            if (allpars.get() == 0) allpars.reset(nll.getParameters((const RooArgSet *)0));
            RooArgSet oldparams(checkpoint->floatParsFinal());
            *allpars = oldparams;
            rStart -= rInc; rInc *= 0.5;
        } else {
            // I did this step, and I'm not there yet
            if (runtimedef::get("FITTER_DYN_STEP")) {
                if (fabs(here - level) > 0.05) { // we still have to step carefully
                    if ((here-there)*(level-there) > 0) { // if we went in the right direction
                        // then optimize step size
                        double rIncFactor = std::max(0.2, std::min(2.0, 0.75*(level-there)/(here-there)));
                        //printf("\t\t\t\t\tCase B1: level-there = %f,  here-there = %f,   rInc(Old) = %f,  rInFactor = %f,  rInc(New) = %f\n", level-there, here-there, rInc, rIncFactor, rInc*rIncFactor);
                        rInc *= rIncFactor;
                    } //else printf("\t\t\t\t\tCase B3: level-there = %f,  here-there = %f,   rInc(Old) = %f\n", level-there, here-there, rInc);
                } else { // close enough to jump straight to target
                    double rIncFactor = std::max(0.05, std::min(4.0, 0.95*(level-there)/(here-there)));
                    //printf("\t\t\t\t\tCase B2: level-there = %f,  here-there = %f,   rInc(Old) = %f,  rInFactor = %f,  rInc(New) = %f\n", level-there, here-there, rInc, rIncFactor, rInc*rIncFactor);
                    rInc *= rIncFactor;
                }
            } else {
                //nothing?
            }
            checkpoint.reset(minim.save());
        }
    } while (fabs(rInc) > minimizerToleranceForMinos_*stepSize_*std::max(1.0,rBound-rStart));
    if (fabs(here - level) > 0.01) {
        std::cout << "Error: closed range without finding crossing." << std::endl;
        return NAN;
    } else {
        return r.getVal();
    }
}
