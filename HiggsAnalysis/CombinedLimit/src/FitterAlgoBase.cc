#include "../interface/FitterAlgoBase.h"
#include <limits>
#include <cmath>

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
#include "../interface/RooMinimizerOpt.h"
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
#include "../interface/ToyMCSamplerOpt.h"

#include "../interface/ProfilingTools.h"


#include <Math/MinimizerOptions.h>
#include <Math/QuantFuncMathCore.h>
#include <Math/ProbFunc.h>

using namespace RooStats;

std::string FitterAlgoBase::minimizerAlgo_ = "Minuit2";
std::string FitterAlgoBase::minimizerAlgoForMinos_ = "Minuit2,simplex";
#include <Math/QuantFuncMathCore.h>
#include <Math/ProbFunc.h>
float       FitterAlgoBase::minimizerTolerance_ = 1e-2;
float       FitterAlgoBase::minimizerToleranceForMinos_ = 1e-4;
int         FitterAlgoBase::minimizerStrategy_  = 1;
int         FitterAlgoBase::minimizerStrategyForMinos_ = 0;
float       FitterAlgoBase::preFitValue_ = 1.0;
float       FitterAlgoBase::stepSize_ = 0.1;
bool        FitterAlgoBase::robustFit_ = false;
int         FitterAlgoBase::maxFailedSteps_ = 5;
bool        FitterAlgoBase::do95_ = false;
bool        FitterAlgoBase::saveNLL_ = false;
bool        FitterAlgoBase::keepFailures_ = false;
bool        FitterAlgoBase::protectUnbinnedChannels_ = false;
float       FitterAlgoBase::nllValue_ = std::numeric_limits<float>::quiet_NaN();

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
        ("saveNLL",  "Save the negative log-likelihood at the minimum in the output tree (note: value is relative to the pre-fit state)")
        ("keepFailures",  "Save the results even if the fit is declared as failed (for NLL studies)")
        ("protectUnbinnedChannels", "Protect PDF from going negative in unbinned channels")
    ;
}

void FitterAlgoBase::applyOptionsBase(const boost::program_options::variables_map &vm) 
{
    saveNLL_ = vm.count("saveNLL");
    keepFailures_ = vm.count("keepFailures");
    protectUnbinnedChannels_ = vm.count("protectUnbinnedChannels");
}

bool FitterAlgoBase::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
  CloseCoutSentry sentry(verbose < 0);

  static bool shouldCreateNLLBranch = saveNLL_;
  if (shouldCreateNLLBranch) { Combine::addBranch("nll", &nllValue_, "nll/F"); shouldCreateNLLBranch = false; }

  RooAbsData *theData = &data;
  std::auto_ptr<toymcoptutils::SimPdfGenInfo> generator;
  std::auto_ptr<RooAbsData> generatedData;
  if (protectUnbinnedChannels_) {
      RooRealVar *weightVar = 0;
      generator.reset(new toymcoptutils::SimPdfGenInfo(*mc_s->GetPdf(), *mc_s->GetObservables(), false));
      generatedData.reset(generator->generateEpsilon(weightVar));
      theData = &*generatedData;
      for (int i = 0, n = data.numEntries(); i < n; ++i) {
        const RooArgSet &entry = *data.get(i);
        theData->add(entry, data.weight());
      }
  }

  bool ret = runSpecific(w, mc_s, mc_b, *theData, limit, limitErr, hint);
  if (protectUnbinnedChannels_) { 
    // destroy things in the proper order
    nll.reset(); // can't keep this
    generatedData.reset(); 
    generator.reset(); 
  }
  return ret;
}


RooFitResult *FitterAlgoBase::doFit(RooAbsPdf &pdf, RooAbsData &data, RooRealVar &r, const RooCmdArg &constrain, bool doHesse, int ndim, bool reuseNLL, bool saveFitResult) {
    return doFit(pdf, data, RooArgList (r), constrain, doHesse, ndim, reuseNLL, saveFitResult);
}

RooFitResult *FitterAlgoBase::doFit(RooAbsPdf &pdf, RooAbsData &data, const RooArgList &rs, const RooCmdArg &constrain, bool doHesse, int ndim, bool reuseNLL, bool saveFitResult) {
    RooFitResult *ret = 0;
    if (reuseNLL && nll.get() != 0) nll->setData(data);	// reuse nll but swap out the data
    else nll.reset(pdf.createNLL(data, constrain, RooFit::Extended(pdf.canBeExtended()))); // make a new nll

    double nll0 = nll->getVal();
    double delta68 = 0.5*ROOT::Math::chisquared_quantile_c(1-0.68,ndim);
    double delta95 = 0.5*ROOT::Math::chisquared_quantile_c(1-0.95,ndim);
    CascadeMinimizer minim(*nll, CascadeMinimizer::Unconstrained, rs.getSize() ? dynamic_cast<RooRealVar*>(rs.first()) : 0);
    minim.setStrategy(minimizerStrategy_);
    minim.setErrorLevel(delta68);
    CloseCoutSentry sentry(verbose < 3);    
    if (verbose>1) std::cout << "do first Minimization " << std::endl;
    TStopwatch tw; 
    if (verbose) tw.Start();
    bool ok = minim.minimize(verbose);
    if (verbose>1) {
       std::cout << "Minimized in : " ; tw.Print();
    }
    nllValue_ =  nll->getVal() - nll0;
    if (!ok && !keepFailures_) { std::cout << "Initial minimization failed. Aborting." << std::endl; return 0; }
    if (doHesse) minim.minimizer().hesse();
    sentry.clear();
    ret = (saveFitResult || rs.getSize() ? minim.save() : new RooFitResult("dummy","success"));
    if (verbose > 1 && ret != 0 && (saveFitResult || rs.getSize())) { ret->Print("V");  }

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
                throw std::runtime_error("95% CL errors with Minos are not working at the moment.");
                minim.setErrorLevel(delta95);
                minim.improve(verbose-1);
                minim.setErrorLevel(delta95);
                if (minim.minos(RooArgSet(r)) != -1) {
                    rf.setRange("err95", r.getVal() + r.getAsymErrorLo(), r.getVal() + r.getAsymErrorHi());
                }
                minim.setErrorLevel(delta68);
                minim.improve(verbose-1);
            }
            if (verbose) std::cout << "Running Minos for POI " << std::endl;
            minim.minimizer().setPrintLevel(2);
            if (verbose>1) {tw.Reset(); tw.Start();}
            if (minim.minos(RooArgSet(r))) {
               if (verbose>1) std::cout << "Run Minos in  "; tw.Print(); std::cout << std::endl;
               rf.setRange("err68", r.getVal() + r.getAsymErrorLo(), r.getVal() + r.getAsymErrorHi());
               rf.setAsymError(r.getAsymErrorLo(), r.getAsymErrorHi());
            }
       } else {
            r.setVal(r0); r.setConstant(true);
            
            CascadeMinimizer minim2(*nll, CascadeMinimizer::Constrained);
            minim2.setStrategy(minimizerStrategyForMinos_);

            std::auto_ptr<RooArgSet> allpars(nll->getParameters((const RooArgSet *)0));

            double nll0 = nll->getVal();
            double threshold68 = nll0 + delta68;
            double threshold95 = nll0 + delta95;
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
    if (runtimedef::get("FITTER_NEW_CROSSING_ALGO")) {
        return findCrossingNew(minim, nll, r, level, rStart, rBound);
    }
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

double FitterAlgoBase::findCrossingNew(CascadeMinimizer &minim, RooAbsReal &nll, RooRealVar &r, double level, double rStart, double rBound) {
    ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgoForMinos_, minimizerToleranceForMinos_);
    CloseCoutSentry sentry(verbose < 3);    

    if (verbose) fprintf(sentry.trueStdOut(), "Searching for crossing at nll = %g in the interval [ %g , %g ]\n", level, rStart, rBound);

    //std::auto_ptr<RooArgSet>    allpars(nll.getParameters((const RooArgSet *)0));
    //utils::CheapValueSnapshot checkpoint(*allpars);
    r.setVal(rStart); 
    if (!minim.improve(verbose-1)) { fprintf(sentry.trueStdOut(), "Error: minimization failed at %s = %g\n", r.GetName(), rStart); return NAN; }
    double quadCorr = 0.0;
    double rVal   = rStart;

    bool unbound = !runtimedef::get("FITTER_BOUND");
    bool neverGiveUp = runtimedef::get("FITTER_NEVER_GIVE_UP");
    double stepSize = stepSize_;
    for (int iter = 0; iter < 20; ++iter) {
        rVal = rStart; r.setVal(rVal); 
        nll.clearEvalErrorLog(); 
        double yStart = nll.getVal();
        if (nll.numEvalErrors() > 0 || std::isnan(yStart) || std::isinf(yStart)) { 
            fprintf(sentry.trueStdOut(), "Error: logEvalErrors on stat of loop for iteration %d, x %+10.6f\n", iter, rVal); return NAN; 
        }
        double rInc = stepSize*(rBound - rStart);
        if (rInc == 0) break;
        if (verbose > 1) fprintf(sentry.trueStdOut(), "x %+10.6f   y %+10.6f                       step %+10.6f [ START OF ITER %d, bound %+10.6f ]\n", rVal, yStart-level, rInc, iter, rBound);  
        // first move w/o profiling
        bool hitbound = true; //, hiterr = false;
        while (unbound || (rBound - rVal - rInc)*rInc >= 0) { // if I've not yet reached the boundary
            rVal += rInc;
            r.setVal(rVal);
            if (r.getVal() != rVal) { fprintf(sentry.trueStdOut(), "Error: can't set %s = %g\n", r.GetName(), rVal); return NAN; }
            nll.clearEvalErrorLog();
            double y = nll.getVal();
            if (nll.numEvalErrors() > 0 || std::isnan(y) || std::isinf(y) || fabs(y-level) > 1e6) { 
                if (verbose > 1) fprintf(sentry.trueStdOut(), "logEvalErrors on stepping for iteration %d, set range to [ %+10.6f, %+10.6f ]\n", iter, rStart, rVal);
                rVal -= rInc; r.setVal(rVal);
                //hiterr = true;
                hitbound = false;
                stepSize *= 0.3; // we have to step very carefully
                //unbound = true; // boundaries are real, so we might have to go very close to them
                break;
            }
            double yCorr = y - quadCorr*std::pow(rVal-rStart,2);
            if (verbose > 1) fprintf(sentry.trueStdOut(), "x %+10.6f   y %+10.6f   yCorr %+10.6f\n", rVal, y-level, yCorr-level);  
            if (fabs(yCorr - yStart) > 0.7) { 
                hitbound = false;
                if (verbose > 1) fprintf(sentry.trueStdOut(), "     --------> accumulated big change in NLL, will go do minimize\n");
                break; 
            }
            if ((level-yCorr)*(level-yStart) < 0) { 
                if (verbose > 1) fprintf(sentry.trueStdOut(), "     --------> found crossing\n");
                double r2 = rVal - rInc; //r2 should be on the same side as yStart, yCorr(rVal) on the opposite
                for (int iter2 = 0; (fabs(yCorr - level) > minimizerTolerance_) && iter2 < 5; ++iter2) {
                    double rMid = 0.5*(rVal+r2); r.setVal(rMid);
                    y = nll.getVal(); yCorr = y - quadCorr*std::pow(rMid-rStart,2);
                    if (verbose > 1) fprintf(sentry.trueStdOut(), "x %+10.6f   y %+10.6f   yCorr %+10.6f   [ bisection iter %d in  %+10.6f in %+10.6f ]\n", rMid, y-level, yCorr-level, iter2, r2, rVal);
                    if ( (level-yCorr)*(level - yStart) < 0 ) {
                        rVal = rMid; // yCorr(rMid) is on same side as yCorr(rVal), so rMid replaces rVal
                    } else {
                        r2   = rMid;
                    }
                }
                r.setVal(rVal); // save final value after bisection loop
                if (verbose > 1) fprintf(sentry.trueStdOut(), "     --------> ending with x %+10.6f\n", rVal);
                hitbound = false; break;
            }
        } 
        // if we have hit an error, rBound has been updated and we just restart stepping slowly towards it
        //if (hiterr) continue;

        // now we profile
        double yUnprof = nll.getVal(), yCorr = yUnprof - quadCorr*std::pow(rVal-rStart,2);
        if (!minim.improve(verbose-1))  { fprintf(sentry.trueStdOut(), "Error: minimization failed at %s = %g\n", r.GetName(), rVal); if (!neverGiveUp) return NAN; }
        double yProf = nll.getVal();
        if (verbose > 1) fprintf(sentry.trueStdOut(), "x %+10.6f   y %+10.6f   yCorr %+10.6f   yProf  %+10.6f   (P-U) %+10.6f    (P-C) %+10.6f    oldSlope %+10.6f    newSlope %+10.6f\n", 
                                                       rVal, yUnprof-level, yCorr-level, yProf-level, yProf - yUnprof, yProf - yCorr, quadCorr, (yUnprof-yProf)/std::pow(rVal-rStart,2));  

        // if on target, return best point from linear interpolation
        if (fabs(yProf - level) < minimizerTolerance_) {
            double w0 = fabs(yProf - level), w1 = fabs(yStart - level);
            return (w1*rVal + w0*rStart)/(w0+w1);
        }

        // save checkpoint
        //checkpoint.readFrom(*allpars);
        // update correction
        if (rVal != rStart) quadCorr = (yUnprof-yProf)/std::pow(rVal-rStart,2);
        if ((level - yStart)*(level - yProf) > 0) {
            // still on the same side as rStart
            rStart = rVal; 
            if (hitbound) {
                fprintf(sentry.trueStdOut(), "Error: closed range at %s = %g without finding any crossing \n", r.GetName(), rVal); 
                return rVal; 
            } else {
                if (verbose > 1) fprintf(sentry.trueStdOut(), " ---> change search window to [ %g , %g ]\n", rStart, rBound);
            }
        } else {
            rBound = rStart;
            rStart = rVal;
            unbound = true; // I did have a bracketing, so I don't need external bounds anymore
            if (verbose > 1) fprintf(sentry.trueStdOut(), " ---> all your brackets are belong to us!!\n");
            if (verbose > 1) fprintf(sentry.trueStdOut(), " ---> change search window to [ %g , %g ]\n", rStart, rBound);
        }
    }
    fprintf(sentry.trueStdOut(), "Error: search did not converge, will return approximate answer %+.6f\n",rVal); 
    return rVal;
}
