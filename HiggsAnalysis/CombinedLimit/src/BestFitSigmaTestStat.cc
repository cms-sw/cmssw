#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/BestFitSigmaTestStat.h"
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
#define DEBUG_FIT_STATUS
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
#define DBG_PLTestStat_ctor 0 // dump parameters in c-tor
#define DBG_PLTestStat_pars 0 // dump parameters in eval
#define DBG_PLTestStat_fit  0 // dump fit result
#define DBG_PLTestStat_main 1 // limited debugging (final NLLs, failed fits)
#else
#define DBG(X,Z) 
#define DBGV(X,Z) 
#endif


//============================================================================================================================================
BestFitSigmaTestStat::BestFitSigmaTestStat(
    const RooArgSet & observables,
    RooAbsPdf &pdf, 
    const RooArgSet *nuisances, 
    const RooArgSet & params,
    int verbosity)
:
    pdf_(&pdf),
    verbosity_(verbosity)
{
    params.snapshot(snap_,false);
    ((RooRealVar*)snap_.find(params.first()->GetName()))->setConstant(false);
    poi_.add(snap_);
    if (nuisances) { nuisances_.add(*nuisances); snap_.addClone(*nuisances, /*silent=*/true); }
    params_.reset(pdf_->getParameters(observables));
}


Double_t BestFitSigmaTestStat::Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/)
{
    // Take snapshot of initial state, to restore it at the end 
    RooArgSet initialState; params_->snapshot(initialState);

   // Initialize parameters
    *params_ = snap_;

    // Initialize signal strength
    RooRealVar *rIn = (RooRealVar *) poi_.first();
    RooRealVar *r   = (RooRealVar *) params_->find(rIn->GetName());
    bool canKeepNLL = createNLL(*pdf_, data);

    double initialR = rIn->getVal();

    // Perform unconstrained minimization
    r->setVal(initialR == 0 ? 0.5 : 0.5*initialR); //best guess
    r->setConstant(false);

    std::cout << "Doing a fit for with " << r->GetName() << " in range [ " << r->getMin() << " , " << r->getMax() << "]" << std::endl;
    std::cout << "Starting point is " << r->GetName() << " = " << r->getVal() << std::endl;
    minNLL(/*constrained=*/false, r);
    double bestFitR = r->getVal();
    std::cout << "Fit result was " << r->GetName() << " = " << r->getVal() << std::endl;

    //Restore initial state, to avoid issues with ToyMCSampler
    *params_ = initialState;

    if (!canKeepNLL) nll_.reset();

    return bestFitR;
}

bool BestFitSigmaTestStat::createNLL(RooAbsPdf &pdf, RooAbsData &data) 
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

double BestFitSigmaTestStat::minNLL(bool constrained, RooRealVar *r) 
{
    CascadeMinimizer::Mode mode(constrained ? CascadeMinimizer::Constrained : CascadeMinimizer::Unconstrained);
    CascadeMinimizer minim(*nll_, mode, r);
    minim.minimize(verbosity_-2);
    return nll_->getVal();
}
