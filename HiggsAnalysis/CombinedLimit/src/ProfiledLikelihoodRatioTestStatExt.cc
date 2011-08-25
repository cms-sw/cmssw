#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
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

#if 0
#define DBG(X,Z) if (X) { Z; }
#define DBGV(X,Z) if (X>1) { Z; }
#define DBG_TestStat_params 0 // ProfiledLikelihoodRatioTestStatOpt::Evaluate; 1 = dump nlls; 2 = dump params at each eval
#define DBG_TestStat_NOFIT  0 // FIXME HACK: if set, don't profile the likelihood, just evaluate it
#define DBG_PLTestStat_ctor 1 // dump parameters in c-tor
#define DBG_PLTestStat_pars 1 // dump parameters in eval
#define DBG_PLTestStat_fit  1 // dump fit result
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
    if (nuisances) nuisances_.addClone(*nuisances);
}

Double_t ProfiledLikelihoodRatioTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    *paramsNull_ = nuisances_;
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
        
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of null pdf (pre fit)" << pdfNull_->GetName()))
    DBGV(DBG_TestStat_params, (paramsNull_->Print("V")))

    bool canKeepNullNLL = createNLL(*pdfNull_, data, nllNull_);
    double nullNLL = minNLL(nllNull_);
    if (!canKeepNullNLL) nllNull_.reset();
    
    *paramsAlt_ = nuisances_;
    *paramsAlt_ = snapAlt_;
        
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of alt pdf " << pdfAlt_->GetName()))
    DBGV(DBG_TestStat_params, (paramsAlt_->Print("V")))
    bool canKeepAltNLL = createNLL(*pdfAlt_, data, nllAlt_);
    double altNLL = minNLL(nllAlt_);
    if (!canKeepAltNLL) nllAlt_.reset();
    
    DBG(DBG_TestStat_params, (printf("Pln: null = %+8.4f, alt = %+8.4f\n", nullNLL, altNLL)))
    return nullNLL-altNLL;
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
    RooMinimizer minim(*nll_);
    minim.setStrategy(0);
    minim.setPrintLevel(verbosity_-2);
    minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
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
    const RooArgList &gobsParams,
    const RooArgList &gobs,
    int verbosity)
:
    pdf_(&pdf),
    gobsParams_(gobsParams),
    gobs_(gobs),
    verbosity_(verbosity)
{
    std::cout << "Created for " << pdf.GetName() << "." << std::endl;
    params.snapshot(snap_,false);
    ((RooRealVar*)snap_.find(params.first()->GetName()))->setConstant(false);
    poi_.add(snap_);
    if (nuisances) { nuisances_.add(*nuisances); snap_.addClone(*nuisances, /*silent=*/true); }
    params_.reset(pdf_->getParameters(observables));
    DBG(DBG_PLTestStat_ctor, (std::cout << "Observables: " << std::endl)) DBG(DBG_PLTestStat_ctor, (observables.Print("V")))
    DBG(DBG_PLTestStat_ctor, (std::cout << "All params: " << std::endl))  DBG(DBG_PLTestStat_ctor, (params_->Print("V")))
    DBG(DBG_PLTestStat_ctor, (std::cout << "Snapshot: " << std::endl))    DBG(DBG_PLTestStat_ctor, (snap_.Print("V")))
    DBG(DBG_PLTestStat_ctor, (std::cout << "POI: " << std::endl))         DBG(DBG_PLTestStat_ctor, (poi_.Print("V")))
}


Double_t ProfiledLikelihoodTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/)
{
    std::cout << "Being evaluated on " << data.GetName() << std::endl;

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

    double initialR = rIn->getVal();

    // Perform unconstrained minimization (denominator)
    r->setMin(0); if (initialR == 0) r->removeMax(); else r->setMax(1.1*initialR); 
    r->setVal(initialR == 0 ? 0.5 : 0.5*initialR); //best guess
    r->setConstant(false);
    DBG(DBG_PLTestStat_pars, (std::cout << "r In: ")) DBG(DBG_PLTestStat_pars, (rIn->Print(""))) DBG(DBG_PLTestStat_pars, std::cout << std::endl)
    DBG(DBG_PLTestStat_pars, std::cout << "r before the fit: ") DBG(DBG_PLTestStat_pars, r->Print("")) DBG(DBG_PLTestStat_pars, std::cout << std::endl)

    double nullNLL = minNLL();
    double bestFitR = r->getVal();

    DBG(DBG_PLTestStat_pars, (std::cout << "r after the fit: ")) DBG(DBG_PLTestStat_pars, (r->Print(""))) DBG(DBG_PLTestStat_pars, std::cout << std::endl)
    DBG(DBG_PLTestStat_pars, std::cout << "Was evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))

    // Prepare for constrained minimization (numerator)
    r->setVal(initialR); 
    r->setConstant(true);
    double thisNLL = nullNLL;
    if (initialR == 0 || bestFitR < initialR) { 
        // must do constrained fit
        thisNLL = minNLL();
        if (thisNLL - nullNLL < -0.02) { 
            printf("  --> constrained fit is better... will repeat unconstrained fit\n");
            r->setConstant(false);
            nullNLL = minNLL();
            bestFitR = r->getVal();
            if (bestFitR > initialR) {
                printf("   after re-fit, signal %7.4f > %7.4f, test statistics will be zero.\n", bestFitR, initialR);
                thisNLL = nullNLL;
            }
        }
    } else {
        printf("   signal fit %7.4f > %7.4f: don't need to compute numerator\n", bestFitR, initialR);
    }

    //Restore initial state, to avoid issues with ToyMCSampler
    *params_ = initialState;

    if (!canKeepNLL) nll_.reset();

    printf("\nNLLs:  num % 10.4f, den % 10.4f (signal %7.4f), test stat % 10.4f\n", thisNLL, nullNLL, bestFitR, thisNLL-nullNLL);
    //return std::min(thisNLL-nullNLL, 0.);
    return thisNLL-nullNLL;
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

double ProfiledLikelihoodTestStatOpt::minNLL() 
{
    RooMinimizer minim(*nll_);
    minim.setPrintLevel(verbosity_-2);  
    minim.setStrategy(0);
    nllutils::robustMinimize(*nll_, minim, verbosity_);
    return nll_->getVal();
}

//============================================================ProfiledLikelihoodRatioTestStatExt
bool nllutils::robustMinimize(RooAbsReal &nll, RooMinimizer &minim, int verbosity) 
{
    double initialNll = nll.getVal();
    std::auto_ptr<RooArgSet> pars;
    bool ret = false;
    for (int tries = 0, maxtries = 4; tries <= maxtries; ++tries) {
        int status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
        //if (verbosity > 1) res->Print("V");
        if (status == 0 && nll.getVal() > initialNll + 0.02) {
            std::auto_ptr<RooFitResult> res(minim.save());
            printf("\n  --> false minimum, status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal());
            if (pars.get() == 0) pars.reset(nll.getParameters((const RooArgSet*)0));
            *pars = res->floatParsInit();
            if (tries == 0) {
                printf("    ----> Doing a re-scan and re-trying\n");
                minim.minimize("Minuit2","Scan");
            } else if (tries == 1) {
                printf("    ----> Re-trying with strategy = 1\n");
                minim.setStrategy(1);
            } else if (tries == 2) {
                printf("    ----> Re-trying with strategy = 2\n");
                minim.setStrategy(2);
            } else  {
                printf("    ----> Last attempt: simplex method \n");
                status = minim.minimize("Minuit2","Simplex");
                if (nll.getVal() < initialNll + 0.02) {
                    printf("\n  --> success: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal());
                    ret = true;
                    break;
                } else {
                    printf("\n  --> final fail: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal());
                    return false;
                }
            }
        } else if (status == 0) {  
            printf("\n  --> success: status %d, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, initialNll, nll.getVal(), initialNll - nll.getVal());
            ret = true;
            break;
        } else if (tries != maxtries) {
            std::auto_ptr<RooFitResult> res(minim.save());
            if (tries > 0 && res->edm() < 0.05*ROOT::Math::MinimizerOptions::DefaultTolerance()) {
                printf("\n  --> acceptable: status %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal());
                ret = true;
                break;
            }
            printf("\n  --> partial fail: status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal());
            if (tries == 1) {
                printf("    ----> Doing a re-scan first, and switching to strategy 1\n");
                minim.minimize("Minuit2","Scan");
                minim.setStrategy(1);
            }
            if (tries == 2) {
                printf("    ----> trying with strategy = 2\n");
                minim.minimize("Minuit2","Scan");
                minim.setStrategy(2);
            }
        } else {
            std::auto_ptr<RooFitResult> res(minim.save());
            printf("\n  --> final fail: status %d, cov. quality %d, edm %10.7f, nll initial % 10.4f, nll final % 10.4f, change %10.5f\n", status, res->covQual(), res->edm(), initialNll, nll.getVal(), initialNll - nll.getVal());
        }
    }
    return ret;
}

/// This below is useless but I like it so I didn't delete it yet
bool 
nllutils::randomWalk(RooAbsReal &nll, RooMinimizer &minimizer, const RooArgSet &parameters, int steps, double scale) 
{
    std::vector<RooRealVar*> vars;  
    std::vector<std::pair<double,double> > ranges;
    vars.reserve(parameters.getSize());
    ranges.reserve(parameters.getSize());
    std::auto_ptr<TIterator> it(parameters.createIterator());
    for (RooAbsArg *a = (RooAbsArg *) it->Next(); a != 0; a = (RooAbsArg *) it->Next()) {
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv != 0 && !rrv->isConstant()) {
            vars.push_back(rrv);
            ranges.push_back(std::pair<double,double>(rrv->getMin(), rrv->getMax()));
        }
    }
    double y0 = nll.getVal(); 
    int nvar = vars.size();
    int acc = 0;
    for (int i = 1, nx = steps / 5; i <= steps; ++i) {
        int j = RooRandom::integer(nvar);
        double dx = (ranges[j].second-ranges[j].first);
        double x0 = vars[j]->getVal(), x1 = x0 + dx*RooRandom::gaussian()*scale;
        while (x1 > ranges[j].second) x1 -= dx;
        while (x1 < ranges[j].first)  x1 += dx;
        vars[j]->setVal(x1);
        double y1 = nll.getVal();
        if (y1 < y0 || exp(y0-y1) > RooRandom::uniform()) {
            y0 = y1;
            acc++;
        } else {
            vars[j]->setVal(x0);
        }
        if (i % nx == 0) scale *= 0.8;
    }
    return (acc > 0.1*steps);
}
