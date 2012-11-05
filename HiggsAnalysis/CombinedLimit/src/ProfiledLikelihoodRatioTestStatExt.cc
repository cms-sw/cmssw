#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/CascadeMinimizer.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"
#include <stdexcept>
#include <RooRealVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooSimultaneous.h>
#include <RooCategory.h>
#include <RooRandom.h>
#include <Math/MinimizerOptions.h>
#include <RooStats/RooStatsUtils.h>
#include "../interface/ProfilingTools.h"

//---- Uncomment this and run with --perfCounters to get statistics of successful and failed fits
//#define DEBUG_FIT_STATUS
#ifdef DEBUG_FIT_STATUS
#define  COUNT_ONE(x) PerfCounter::add(x);
#else
#define  COUNT_ONE(X) 
#endif

//---- Uncomment this and set some of these to 1 to get more debugging
#if 0
#define DBG(X,Z) if (X) { Z; }
#define DBGV(X,Z) if (X>1) { Z; }
#define DBG_TestStat_params 0 // ProfiledLikelihoodRatioTestStatOpt::Evaluate; 1 = dump nlls; 2 = dump params at each eval
#define DBG_TestStat_NOFIT  0 // FIXME HACK: if set, don't profile the likelihood, just evaluate it
#define DBG_PLTestStat_ctor 1 // dump parameters in c-tor
#define DBG_PLTestStat_pars 1 // dump parameters in eval
#define DBG_PLTestStat_fit  1 // dump fit result
#define DBG_PLTestStat_main 1 // limited debugging (final NLLs, failed fits)
#else
#define DBG(X,Z) 
#define DBGV(X,Z) 
#endif

//============================================================================================================================================
ProfiledLikelihoodRatioTestStatOpt::ProfiledLikelihoodRatioTestStatOpt(
    const RooArgSet & observables,
    RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt,
    const RooArgSet *nuisances,
    const RooArgSet & paramsNull, const RooArgSet & paramsAlt,
    int verbosity)
: 
    pdfNull_(&pdfNull), pdfAlt_(&pdfAlt),
    paramsNull_(pdfNull_->getVariables()), 
    paramsAlt_(pdfAlt_->getVariables()), 
    verbosity_(verbosity)
{
    snapNull_.addClone(paramsNull);
    snapAlt_.addClone(paramsAlt);
    DBG(DBG_TestStat_params, (std::cout << "Null snapshot" << pdfNull_->GetName() << "\n")) DBG(DBG_TestStat_params, (snapNull_.Print("V")))
    DBG(DBG_TestStat_params, (std::cout << "Alt  snapshot" << pdfAlt_->GetName() << "\n"))  DBG(DBG_TestStat_params, (snapAlt_.Print("V")))
    if (nuisances) {
        nuisances_.addClone(*nuisances);
        DBG(DBG_TestStat_params, (std::cout << "Nuisances" << std::endl))  DBG(DBG_TestStat_params, (nuisances_.Print("V")))
    }
}

Double_t ProfiledLikelihoodRatioTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    RooArgSet initialStateAlt;  paramsAlt_->snapshot(initialStateAlt);
    RooArgSet initialStateNull; paramsNull_->snapshot(initialStateNull);

    *paramsNull_ = nuisances_;
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
        
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of null pdf (pre fit)" << pdfNull_->GetName() << "\n"))
    DBGV(DBG_TestStat_params, (paramsNull_->Print("V")))

    bool canKeepNullNLL = createNLL(*pdfNull_, data, nllNull_);
    double nullNLL = minNLL(nllNull_);
    if (!canKeepNullNLL) nllNull_.reset();
    
    *paramsAlt_ = nuisances_;
    *paramsAlt_ = snapAlt_;
        
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of alt pdf " << pdfAlt_->GetName() << "\n"))
    DBGV(DBG_TestStat_params, (paramsAlt_->Print("V")))
    bool canKeepAltNLL = createNLL(*pdfAlt_, data, nllAlt_);
    double altNLL = minNLL(nllAlt_);
    if (!canKeepAltNLL) nllAlt_.reset();
    
    DBG(DBG_TestStat_params, (printf("Pln: null = %+8.4f, alt = %+8.4f\n", nullNLL, altNLL)))
    double ret = nullNLL-altNLL;

    *paramsAlt_  = initialStateAlt;
    *paramsNull_ = initialStateNull;

    return ret;
}

bool ProfiledLikelihoodRatioTestStatOpt::createNLL(RooAbsPdf &pdf, RooAbsData &data, std::auto_ptr<RooAbsReal> &nll_) 
{
    if (typeid(pdf) == typeid(RooSimultaneousOpt)) {
        if (nll_.get() == 0) nll_.reset(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
        else ((cacheutils::CachingSimNLL&)(*nll_)).setData(data);
        return true;
    } else {
        nll_.reset(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
        return false;
    }
}


double ProfiledLikelihoodRatioTestStatOpt::minNLL(std::auto_ptr<RooAbsReal> &nll_) 
{
#if defined(DBG_TestStat_NOFIT) && (DBG_TestStat_NOFIT > 0)
    if (verbosity_ > 0) std::cout << "Profiling likelihood for pdf " << pdf.GetName() << std::endl;
    std::auto_ptr<RooAbsReal> nll_(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
    return nll_->getVal();
#endif
    CascadeMinimizer minim(*nll_, CascadeMinimizer::Constrained);
    minim.setStrategy(0);
    minim.minimize(verbosity_-2);
    if (verbosity_ > 1) {
        std::auto_ptr<RooFitResult> res(minim.save());
        res->Print("V");
    }
    return nll_->getVal();
}

//============================================================================================================================================
ProfiledLikelihoodTestStatOpt::ProfiledLikelihoodTestStatOpt(
    const RooArgSet & observables,
    RooAbsPdf &pdf, 
    const RooArgSet *nuisances, 
    const RooArgSet & params,
    const RooArgSet & poi,
    const RooArgList &gobsParams,
    const RooArgList &gobs,
    int verbosity, 
    OneSidedness oneSided)
:
    pdf_(&pdf),
    gobsParams_(gobsParams),
    gobs_(gobs),
    verbosity_(verbosity),
    oneSided_(oneSided)
{
    DBG(DBG_PLTestStat_main, (std::cout << "Created for " << pdf.GetName() << "." << std::endl))

    params.snapshot(snap_,false);
    ((RooRealVar*)snap_.find(params.first()->GetName()))->setConstant(false);
    if (nuisances) { nuisances_.add(*nuisances); snap_.addClone(*nuisances, /*silent=*/true); }
    params_.reset(pdf_->getParameters(observables));
    DBG(DBG_PLTestStat_ctor, (std::cout << "Observables: " << std::endl)) DBG(DBG_PLTestStat_ctor, (observables.Print("V")))
    DBG(DBG_PLTestStat_ctor, (std::cout << "All params: " << std::endl))  DBG(DBG_PLTestStat_ctor, (params_->Print("V")))
    DBG(DBG_PLTestStat_ctor, (std::cout << "Snapshot: " << std::endl))    DBG(DBG_PLTestStat_ctor, (snap_.Print("V")))
    DBG(DBG_PLTestStat_ctor, (std::cout << "POI: " << std::endl))         DBG(DBG_PLTestStat_ctor, (poi_.Print("V")))
    RooLinkedListIter it = poi.iterator();
    for (RooAbsArg *a = (RooAbsArg*) it.Next(); a != 0; a = (RooAbsArg*) it.Next()) {
        // search for this poi in the parameters and in the snapshot
        RooAbsArg *ps = snap_.find(a->GetName());   
        RooAbsArg *pp = params_->find(a->GetName());
        if (pp == 0) { std::cerr << "WARNING: NLL does not depend on POI " << a->GetName() << ", cannot profile" << std::endl; continue; }
        if (ps == 0) { std::cerr << "WARNING: no snapshot for POI " << a->GetName() << ", cannot profile"  << std::endl; continue; }
        poi_.add(*ps);
        poiParams_.add(*pp);
    }
}


Double_t ProfiledLikelihoodTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/)
{
    bool do_debug = runtimedef::get("DEBUG_PLTSO");
    //static bool do_rescan = runtimedef::get("PLTSO_FULL_RESCAN");
    DBG(DBG_PLTestStat_main, (std::cout << "Being evaluated on " << data.GetName() << std::endl))

    // Take snapshot of initial state, to restore it at the end 
    RooArgSet initialState; params_->snapshot(initialState);

    DBG(DBG_PLTestStat_pars, std::cout << "Being evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))
    // Initialize parameters
    *params_ = snap_;

    //set value of "globalConstrained" nuisances to the value of the corresponding global observable and set them constant
    for (int i=0; i<gobsParams_.getSize(); ++i) {
      RooRealVar *nuis = (RooRealVar*)gobsParams_.at(i);
      RooRealVar *gobs = (RooRealVar*)gobs_.at(i);
      nuis->setVal(gobs->getVal());
      nuis->setConstant();
    }
        
    DBG(DBG_PLTestStat_pars, std::cout << "Being evaluated on " << data.GetName() << ": params after snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))
    // Initialize signal strength (or the first parameter)
    RooRealVar *rIn = (RooRealVar *) poi_.first();
    RooRealVar *r   = (RooRealVar *) params_->find(rIn->GetName());
    bool canKeepNLL = createNLL(*pdf_, data);

    double initialR = rIn->getVal();

    // Perform unconstrained minimization (denominator)
    if (poi_.getSize() == 1) {
        double oldMax = r->getMax();
        if (oneSided_ != signFlipDef ) r->setMin(0); 
        if (initialR == 0 || (oneSided_ != oneSidedDef)) r->removeMax(); else r->setMax(1.1*initialR); 
        r->setVal(initialR == 0 ? (std::isnormal(oldMax) && fabs(oldMax) < 1e28 ? 0.1*oldMax : 0.5) : 0.5*initialR); //best guess
        r->setConstant(false);
    } else {
        utils::setAllConstant(poiParams_,false);
    }
    DBG(DBG_PLTestStat_pars, (std::cout << "r In: ")) DBG(DBG_PLTestStat_pars, (rIn->Print(""))) DBG(DBG_PLTestStat_pars, std::cout << std::endl)
    DBG(DBG_PLTestStat_pars, std::cout << "r before the fit: ") DBG(DBG_PLTestStat_pars, r->Print("")) DBG(DBG_PLTestStat_pars, std::cout << std::endl)

    //std::cout << "PERFORMING UNCONSTRAINED FIT " << r->GetName() << " [ " << r->getMin() << " - " << r->getMax() << " ] "<< std::endl;
    double nullNLL = minNLL(/*constrained=*/false, r);
    double bestFitR = r->getVal();

    DBG(DBG_PLTestStat_pars, (std::cout << "r after the fit: ")) DBG(DBG_PLTestStat_pars, (r->Print(""))) DBG(DBG_PLTestStat_pars, std::cout << std::endl)
    DBG(DBG_PLTestStat_pars, std::cout << "Was evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))

    // Prepare for constrained minimization (numerator)
    if (poi_.getSize() == 1) {
        r->setVal(initialR); 
        r->setConstant(true);
    } else {
        poiParams_.assignValueOnly(poi_);
        utils::setAllConstant(poiParams_,true);
    }
    double thisNLL = nullNLL;
    if (initialR == 0 || oneSided_ != oneSidedDef || bestFitR < initialR) { 
        // must do constrained fit (if there's something to fit besides XS)
        //std::cout << "PERFORMING CONSTRAINED FIT " << r->GetName() << " == " << r->getVal() << std::endl;
        thisNLL = (nuisances_.getSize() > 0 ? minNLL(/*constrained=*/true, r) : nll_->getVal());
        if (thisNLL - nullNLL < -0.02) { 
            DBG(DBG_PLTestStat_main, (printf("  --> constrained fit is better... will repeat unconstrained fit\n")))
            utils::setAllConstant(poiParams_,false);
            nullNLL = minNLL(/*constrained=*/false, r);
            bestFitR = r->getVal();
            if (bestFitR > initialR && oneSided_ == oneSidedDef) {
                DBG(DBG_PLTestStat_main, (printf("   after re-fit, signal %7.4f > %7.4f, test statistics will be zero.\n", bestFitR, initialR)))
                thisNLL = nullNLL;
            }
        }
        /* This piece of debug code was added to see if we were finding a local minimum at zero instead of the global minimum.
           It doesn't seem to be the case, however  
        if (do_rescan && fabs(thisNLL - nullNLL) < 0.2 && initialR == 0) {
            if (do_debug) printf("Doing full rescan. best fit r = %.4f, -lnQ = %.5f\n", bestFitR, thisNLL-nullNLL);
            for (double rx = 0; rx <= 5; rx += 0.01) {
                r->setVal(rx); double y = nll_->getVal();
                if (y < nullNLL) {
                    printf("%4.2f\t%8.5f\t<<==== ALERT\n", rx, y - nullNLL);
                } else {
                    printf("%4.2f\t%8.5f\n", rx, y - nullNLL);
                }
            }
        } */
        if (initialR == 0) { // NOTE: signs are flipped for the zero case!
            if (oneSided_ == signFlipDef && bestFitR < initialR) {
                DBG(DBG_PLTestStat_main, (printf("   fitted signal %7.4f is negative, discovery test statistics will be negative.\n", bestFitR)))
                std::swap(thisNLL, nullNLL);
            }
        } else if (bestFitR > initialR && oneSided_ == signFlipDef) {
            DBG(DBG_PLTestStat_main, (printf("   fitted signal %7.4f > %7.4f, test statistics will be negative.\n", bestFitR, initialR)))
            std::swap(thisNLL, nullNLL);
        }
    } else {
        DBG(DBG_PLTestStat_main, (printf("   signal fit %7.4f > %7.4f: don't need to compute numerator\n", bestFitR, initialR)))
    }

    //Restore initial state, to avoid issues with ToyMCSampler
    *params_ = initialState;

    if (!canKeepNLL) nll_.reset();

    DBG(DBG_PLTestStat_main, (printf("\nNLLs:  num % 10.4f, den % 10.4f (signal %7.4f), test stat % 10.4f\n", thisNLL, nullNLL, bestFitR, thisNLL-nullNLL)))
    if (do_debug) printf("\nNLLs:  num % 10.4f, den % 10.4f (signal %7.4f), test stat % 10.4f\n", thisNLL, nullNLL, bestFitR, thisNLL-nullNLL);
    //return std::min(thisNLL-nullNLL, 0.);
    return thisNLL-nullNLL;
}

std::vector<Double_t> ProfiledLikelihoodTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/, const std::vector<Double_t> &rVals)
{
    static bool do_debug = runtimedef::get("DEBUG_PLTSO");
    std::vector<Double_t> ret(rVals.size(), 0);

    DBG(DBG_PLTestStat_main, (std::cout << "Being evaluated on " << data.GetName() << std::endl))

    // Take snapshot of initial state, to restore it at the end 
    RooArgSet initialState; params_->snapshot(initialState);

    DBG(DBG_PLTestStat_pars, std::cout << "Being evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))
    // Initialize parameters
    *params_ = snap_;

    //set value of "globalConstrained" nuisances to the value of the corresponding global observable and set them constant
    for (int i=0; i<gobsParams_.getSize(); ++i) {
      RooRealVar *nuis = (RooRealVar*)gobsParams_.at(i);
      RooRealVar *gobs = (RooRealVar*)gobs_.at(i);
      nuis->setVal(gobs->getVal());
      nuis->setConstant();
    }
        
    DBG(DBG_PLTestStat_pars, std::cout << "Being evaluated on " << data.GetName() << ": params after snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))
    // Initialize signal strength
    RooRealVar *rIn = (RooRealVar *) poi_.first();
    RooRealVar *r   = (RooRealVar *) params_->find(rIn->GetName());
    bool canKeepNLL = createNLL(*pdf_, data);

    double initialR = rVals.back();

    // Perform unconstrained minimization (denominator)
    r->setMin(0); if (initialR == 0 || (oneSided_ != oneSidedDef)) r->removeMax(); else r->setMax(1.1*initialR); 
    r->setVal(initialR == 0 ? 0.5 : 0.5*initialR); //best guess
    r->setConstant(false);
    DBG(DBG_PLTestStat_pars, (std::cout << "r In: ")) DBG(DBG_PLTestStat_pars, (rIn->Print(""))) DBG(DBG_PLTestStat_pars, std::cout << std::endl)
    DBG(DBG_PLTestStat_pars, std::cout << "r before the fit: ") DBG(DBG_PLTestStat_pars, r->Print("")) DBG(DBG_PLTestStat_pars, std::cout << std::endl)

    if (do_debug) std::cout << "PERFORMING UNCONSTRAINED FIT " << r->GetName() << " [ " << r->getMin() << " - " << r->getVal() << " - " << r->getMax() << " ] "<< std::endl;
    double nullNLL = minNLL(/*constrained=*/false, r);
    double bestFitR = r->getVal();
    // Take snapshot of initial state, to restore it at the end 
    RooArgSet bestFitState; params_->snapshot(bestFitState);


    DBG(DBG_PLTestStat_pars, (std::cout << "r after the fit: ")) DBG(DBG_PLTestStat_pars, (r->Print(""))) DBG(DBG_PLTestStat_pars, std::cout << std::endl)
    DBG(DBG_PLTestStat_pars, std::cout << "Was evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))

    double EPS = 0.25*ROOT::Math::MinimizerOptions::DefaultTolerance();
    for (int iR = 0, nR = rVals.size(); iR < nR; ++iR) {
        if (fabs(ret[iR]) > 10*EPS) continue; // don't bother re-update points which were too far from zero anyway.
        *params_ = bestFitState;
        initialR = rVals[iR];
        // Prepare for constrained minimization (numerator)
        r->setVal(initialR); 
        r->setConstant(true);
        double thisNLL = nullNLL, sign = +1.0;
        if (initialR == 0 || oneSided_ != oneSidedDef || bestFitR < initialR) { 
            // must do constrained fit (if there's something to fit besides XS)
            //std::cout << "PERFORMING CONSTRAINED FIT " << r->GetName() << " == " << r->getVal() << std::endl;
            thisNLL = (nuisances_.getSize() > 0 ? minNLL(/*constrained=*/true, r) : nll_->getVal());
            if (thisNLL - nullNLL < 0 && thisNLL - nullNLL >= -EPS) {
                thisNLL = nullNLL;
            } else if (thisNLL - nullNLL < 0) {
                DBG(DBG_PLTestStat_main, (printf("  --> constrained fit is better... will repeat unconstrained fit\n")))
                r->setConstant(false);
                r->setVal(bestFitR);
                double oldNullNLL = nullNLL;
                nullNLL = minNLL(/*constrained=*/false, r);
                bestFitR = r->getVal();
                bestFitState.removeAll(); params_->snapshot(bestFitState);
                for (int iR2 = 0; iR2 < iR; ++iR2) {
                    ret[iR2] -= (nullNLL - oldNullNLL); // fixup already computed test statistics
                }
                iR = -1; continue; // restart over again, refitting those close to zero :-(
            }
            if (bestFitR > initialR && oneSided_ == signFlipDef) {
                DBG(DBG_PLTestStat_main, (printf("   fitted signal %7.4f > %7.4f, test statistics will be negative.\n", bestFitR, initialR)))
                sign = -1.0;
            }
        } else {
            DBG(DBG_PLTestStat_main, (printf("   signal fit %7.4f > %7.4f: don't need to compute numerator\n", bestFitR, initialR)))
        }

        ret[iR] = sign * (thisNLL-nullNLL);
        DBG(DBG_PLTestStat_main, (printf("\nNLLs for %7.4f:  num % 10.4f, den % 10.4f (signal %7.4f), test stat % 10.4f\n", initialR, thisNLL, nullNLL, bestFitR, ret[iR])))
        if (do_debug) { 
            printf("   Q(%.4f) = %.4f (best fit signal %.4f), from num %.4f, den %.4f\n", rVals[iR], ret[iR], bestFitR, thisNLL, nullNLL); fflush(stdout);
        }
    }
    //Restore initial state, to avoid issues with ToyMCSampler
    *params_ = initialState;

    if (!canKeepNLL) nll_.reset();

    //return std::min(thisNLL-nullNLL, 0.);
    return ret;
}
bool ProfiledLikelihoodTestStatOpt::createNLL(RooAbsPdf &pdf, RooAbsData &data) 
{
    if (typeid(pdf) == typeid(RooSimultaneousOpt)) {
        if (nll_.get() == 0) nll_.reset(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
        else ((cacheutils::CachingSimNLL&)(*nll_)).setData(data);
        return true;
    } else {
        nll_.reset(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
        return false;
    }
}

double ProfiledLikelihoodTestStatOpt::minNLL(bool constrained, RooRealVar *r) 
{
    CascadeMinimizer::Mode mode(constrained ? CascadeMinimizer::Constrained : CascadeMinimizer::Unconstrained);
    CascadeMinimizer minim(*nll_, mode, r);
    minim.minimize(verbosity_-2);
    return nll_->getVal();
}

//============================================================ProfiledLikelihoodRatioTestStatExt
bool nllutils::robustMinimize(RooAbsReal &nll, RooMinimizerOpt &minim, int verbosity) 
{
    static bool do_debug = (runtimedef::get("DEBUG_MINIM") || runtimedef::get("DEBUG_PLTSO") > 1);
    double initialNll = nll.getVal();
    std::auto_ptr<RooArgSet> pars;
    bool ret = false;
    for (int tries = 0, maxtries = 4; tries <= maxtries; ++tries) {
        int status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
        //if (verbosity > 1) res->Print("V");
        if (status == 0 && nll.getVal() > initialNll + 0.02) {
            std::auto_ptr<RooFitResult> res(minim.save());
            //PerfCounter::add("Minimizer.save() called for false minimum"); 
            DBG(DBG_PLTestStat_main, (printf("\n  --> false minimum, status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal())))
            if (pars.get() == 0) pars.reset(nll.getParameters((const RooArgSet*)0));
            *pars = res->floatParsInit();
            if (tries == 0) {
                COUNT_ONE("nllutils::robustMinimize: false minimum (first try)")
                DBG(DBG_PLTestStat_main, (printf("    ----> Doing a re-scan and re-trying\n")))
                minim.minimize("Minuit2","Scan");
            } else if (tries == 1) {
                COUNT_ONE("nllutils::robustMinimize: false minimum (second try)")
                DBG(DBG_PLTestStat_main, (printf("    ----> Re-trying with strategy = 1\n")))
                minim.setStrategy(1);
            } else if (tries == 2) {
                COUNT_ONE("nllutils::robustMinimize: false minimum (third try)")
                DBG(DBG_PLTestStat_main, (printf("    ----> Re-trying with strategy = 2\n")))
                minim.setStrategy(2);
            } else  {
                COUNT_ONE("nllutils::robustMinimize: false minimum (third try)")
                DBG(DBG_PLTestStat_main, (printf("    ----> Last attempt: simplex method \n")))
                status = minim.minimize("Minuit2","Simplex");
                if (nll.getVal() < initialNll + 0.02) {
                    DBG(DBG_PLTestStat_main, (printf("\n  --> success: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal())))
                    if (do_debug) printf("\n  --> success: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal());
                    COUNT_ONE("nllutils::robustMinimize: final success")
                    ret = true;
                    break;
                } else {
                    COUNT_ONE("nllutils::robustMinimize: final fail")
                    DBG(DBG_PLTestStat_main, (printf("\n  --> final fail: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal())))
                    if (do_debug) printf("\n  --> final fail: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal());
                    return false;
                }
            }
        } else if (status == 0) {  
            DBG(DBG_PLTestStat_main, (printf("\n  --> success: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal())))
            if (do_debug) printf("\n  --> success: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal());
            COUNT_ONE("nllutils::robustMinimize: final success")
            ret = true;
            break;
        } else if (tries != maxtries) {
            std::auto_ptr<RooFitResult> res(do_debug ? minim.save() : 0);
            //PerfCounter::add("Minimizer.save() called for failed minimization"); 
            if (tries > 0 && minim.edm() < 0.05*ROOT::Math::MinimizerOptions::DefaultTolerance()) {
                DBG(DBG_PLTestStat_main, (printf("\n  --> acceptable: status %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal())))
                if (do_debug) printf("\n  --> acceptable: status %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal());
                COUNT_ONE("nllutils::robustMinimize: accepting fit with bad status but good EDM")
                COUNT_ONE("nllutils::robustMinimize: final success")
                ret = true;
                break;
            }
            DBG(DBG_PLTestStat_main, (printf("\n  --> partial fail: status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal())))
            if (tries == 1) {
                COUNT_ONE("nllutils::robustMinimize: failed first attempt")
                DBG(DBG_PLTestStat_main, (printf("    ----> Doing a re-scan first, and switching to strategy 1\n")))
                minim.minimize("Minuit2","Scan");
                minim.setStrategy(1);
            }
            if (tries == 2) {
                COUNT_ONE("nllutils::robustMinimize: failed second attempt")
                DBG(DBG_PLTestStat_main, (printf("    ----> trying with strategy = 2\n")))
                minim.minimize("Minuit2","Scan");
                minim.setStrategy(2);
            }
        } else {
            std::auto_ptr<RooFitResult> res(do_debug ? minim.save() : 0);
            DBG(DBG_PLTestStat_main, (printf("\n  --> final fail: status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal())))
            if (do_debug) printf("\n  --> final fail: status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal());
            COUNT_ONE("nllutils::robustMinimize: final fail")
        }
    }
    return ret;
}

