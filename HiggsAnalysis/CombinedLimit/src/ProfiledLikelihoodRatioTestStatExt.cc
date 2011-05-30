#include "HiggsAnalysis/CombinedLimit/interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "HiggsAnalysis/CombinedLimit/interface/CloseCoutSentry.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"
#include <stdexcept>
#include <RooRealVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <Math/MinimizerOptions.h>
#include <RooStats/RooStatsUtils.h>

#if 0
#define DBG(X,Z) if (X) { Z; }
#define DBGV(X,Z) if (X>1) { Z; }
#define DBG_DataNLL_ctor 0  //OptimizedDatasetForNLL::OptimizedDatasetForNLL (once)
#define DBG_DataNLL_add  0  //OptimizedDatasetForNLL::addEntries (once per data entry in sim pdf)
#define DBG_DataNLL_get  0  //OptimizedDatasetForNLL::get        (once per data entry per eval)
#define DBG_SimpleNLL_add    0   //OptimizedSimpleNLL::addData (once per data entry in sim pdfs)
#define DBG_SimpleNLL_eval   0   //OptimizedSimpleNLL::evaluate (1 = once per call, 2 = once per entry per call)
#define DBG_SimNLL_ctor 0 // OptimizedSimNLL::init factorization of pdf (once)
#define DBG_SimNLL_setD 0 // OptimizedSimNLL::init factorization of pdf (1 = once, 2 = once per entry)
#define DBG_SimNLL_eval  0 // OptimizedSimNLL::init factorization of pdf (1 = once, 2 = once per param)
#define DBG_SimNLL_cache 0 // OptimizedSimNLL::init factorization of pdf (once per eval)
#define DBG_TestStat_params 0 // ProfiledLikelihoodRatioTestStatExt::Evaluate; 1 = dump nlls; 2 = dump params at each eval
#define DBG_TestStat_NOFIT  0 // FIXME HACK: if set, don't profile the likelihood, just evaluate it
#define DBG_PLTestStat_ctor 0 // dump parameters in c-tor
#define DBG_PLTestStat_pars 0 // dump parameters in eval
#define DBG_PLTestStat_fit  0 // dump fit result
#else
#define DBG(X,Z) 
#define DBGV(X,Z) 
#endif
//============================================================================================================================================
//============================================================================================================================================
OptimizedDatasetForNLL::OptimizedDatasetForNLL(const RooArgSet &vars, const RooAbsData *data, int firstEntry, int lastEntry) :
    numEntries_(0),
    sumw_(0),
    obs_(vars)
{
    DBG(DBG_DataNLL_ctor, (std::cout << "Creating dataset for nll from " << data->GetName() << ", range " << firstEntry << "-" << (numEntries_-firstEntry) << std::endl))
    // instantiate vars
    vars_.resize(obs_.getSize());
    std::auto_ptr<TIterator> iter(vars.createIterator());
    RooAbsArg *arg = (RooAbsArg*) iter->Next();
    int i = 0;
    for (i = 0; arg != 0; ++i, arg = (RooAbsArg*) iter->Next()) {
        vars_[i].var = dynamic_cast<RooRealVar *>(arg);
        if (vars_[i].var == 0) { --i; continue; }
        vars_[i].vals.reserve((lastEntry == -1 ? data->numEntries() : lastEntry) - firstEntry);
        DBG(DBG_DataNLL_ctor,(std::cout << "  attached var " << vars_[i].var->GetName() << std::endl))
    }
    vars_.resize(i); 
    DBG(DBG_DataNLL_ctor,(std::cout << "  total attached vars " << vars_.size() << std::endl))
    weight_.reserve((lastEntry == -1 ? data->numEntries() : lastEntry) - firstEntry);
    dataset_ = 0; datasetVars_.clear();
    addEntries(data, firstEntry, lastEntry);
}

void OptimizedDatasetForNLL::addEntries(const RooAbsData *data, int firstEntry, int lastEntry)
{
    int addedEntries = ((lastEntry == -1 ? data->numEntries() : lastEntry) - firstEntry);
    numEntries_ += addedEntries;
    assert(addedEntries > 0 && "Can't add zero entries to an OptimizedDatasetForNLL for zero entries");
    if (data != dataset_) {
        dataset_ = data;
        // get dataset vars
        const RooArgSet *dataEntry = data->get();
        datasetVars_.resize(vars_.size()); 
        for (int i = 0, nv = vars_.size(); i < nv; ++i) {
            datasetVars_[i] =  dynamic_cast<RooRealVar *>(dataEntry->find(vars_[i].var->GetName()));
            assert(datasetVars_[i] != 0 && "Observable not in dataset");
        }
    }
    // fill vars
    std::vector<Var>::iterator itv, bv = vars_.begin(), ev = vars_.end();
    std::vector<RooRealVar *>::const_iterator itdv, bdv = datasetVars_.begin();
    for (int j = 0; j < addedEntries; ++j) {
        data->get(j + firstEntry); 
        DBG(DBG_DataNLL_add,(std::cout << "  entry " << j << ": "));
        for (itv = bv, itdv = bdv; itv != ev; ++itv) { 
            itv->vals.push_back((*itdv)->getVal()); 
            DBG(DBG_DataNLL_add,(std::cout << "  " << itv->var->GetName() << "=" << itv->vals[j]))
        }
        double w = data->weight();
        sumw_ += w;
        weight_.push_back(w);
        DBG(DBG_DataNLL_add,(std::cout << std::endl))
    }
    DBG(DBG_DataNLL_add,(std::cout << "  filled vars and weights, total entries = " << weight_.size() << std::endl))
}

OptimizedDatasetForNLL::~OptimizedDatasetForNLL() { }

double OptimizedDatasetForNLL::get(int i) const {
    DBG(DBG_DataNLL_get, (std::cout << "  entry " << i << ": "))
    std::vector<Var>::iterator itv, bv = vars_.begin(), ev = vars_.end();
    for (itv = bv; itv != ev; ++itv) {
        itv->var->setVal(itv->vals[i]);
        DBG(DBG_DataNLL_get, (std::cout << "  " << itv->var->GetName() << "=" << itv->vals[i]))
    }
    DBG(DBG_DataNLL_get,(std::cout << std::endl))
    return weight_[i];
}


//============================================================================================================================================
//============================================================================================================================================
OptimizedSimpleNLL::OptimizedSimpleNLL(RooAbsPdf *pdf) :
    RooAbsReal(TString("nllSimple_")+pdf->GetName(), ""),
    pdf_(pdf),
    params_("params","parameters",this)
{
}

OptimizedSimpleNLL::OptimizedSimpleNLL(const OptimizedSimpleNLL &other, const char *name) :
    RooAbsReal(name ? name : (TString("nllSimple_")+other.pdf_->GetName()).Data(), ""),
    pdf_(other.pdf_),
    params_("params",this,other.params_)
{
    setValueDirty();
}


void OptimizedSimpleNLL::setData(const RooAbsData &data, int firstEntry, int lastEntry) 
{
    data_.reset();
    addData(data, firstEntry, lastEntry);
}
void OptimizedSimpleNLL::addData(const RooAbsData &data, int firstEntry, int lastEntry) 
{
    // if first time, get observables
    if (obs_.getSize() == 0) {
        DBG(DBG_SimpleNLL_add, (std::cout << "First call to OptimizedSimpleNLL::setData for pdf " << pdf_->GetName() << std::endl));
        std::auto_ptr<RooArgSet> obs(pdf_->getObservables(data));
        obs_.add(*obs);
        DBG(DBG_SimpleNLL_add, (std::cout << "Observables: ")) DBG(DBG_SimpleNLL_add, (obs_.Print("V")))
        std::auto_ptr<RooArgSet> params(pdf_->getParameters(data));
        params_.add(*params);
        pdf_->optimizeCacheMode(*obs);
    }
    // cache dataset
    if (data_.get() == 0) {
        // redirect observable nodes to cache
        data_.reset(new OptimizedDatasetForNLL(obs_, &data, firstEntry, lastEntry));
        std::auto_ptr<TIterator> iter(obs_.createIterator());
        for(RooAbsArg *arg = (RooAbsArg*) iter->Next(); arg != 0; arg = (RooAbsArg*) iter->Next()) {
            DBG(DBG_SimpleNLL_add, (std::cout << "  redirected servers for " << arg->GetName() << std::endl))
            arg->recursiveRedirectServers(data_->obs());
        }
    } else {
        data_->addEntries(&data, firstEntry, lastEntry);
    }
    setValueDirty();
}

Double_t OptimizedSimpleNLL::evaluate() const {
    DBG(DBG_SimpleNLL_eval, (std::cout << "Called OptimizedSimpleNLL::evaluate() for pdf = " << pdf_->GetName() << std::endl))
    double sum = 0;
    if (data_.get()) {
        OptimizedDatasetForNLL &data = *data_;
        for (int i = 0, n = data.numEntries(); i < n; ++i) {
            RooAbsArg::setDirtyInhibit(true); // avoid propagating ValueDirty for each event
            double weight = data.get(i);
            RooAbsArg::setDirtyInhibit(false); // avoid propagating ValueDirty for each event
            if (weight == 0) continue;
            DBGV(DBG_SimpleNLL_eval, (data_->obs().Print("V")))
            DBGV(DBG_SimpleNLL_eval, (std::cout << "  entry " << i << ", weight " << weight << ", glv = " <<  pdf_->getLogVal(&obs_) << std::endl))
            sum  -= weight * pdf_->getLogVal(&obs_);
        }
        sum += pdf_->extendedTerm(UInt_t(data.sumWeights()), &obs_);
    } else {
        sum += pdf_->extendedTerm(0, &obs_);
    }
    DBG(DBG_SimpleNLL_eval, (std::cout << "total sum(nll) = " << sum << ", sum(w) = " << data_->sumWeights() << ", ext term " << pdf_->extendedTerm(UInt_t(data_->sumWeights()), &obs_) << std::endl))
    return sum;
}
RooArgSet* OptimizedSimpleNLL::getObservables(const RooArgSet* depList, Bool_t valueOnly) const 
{ 
    return new RooArgSet(); 
}
RooArgSet* OptimizedSimpleNLL::getParameters(const RooArgSet* depList, Bool_t stripDisconnected) const 
{ 
    return new RooArgSet(params_); 
}


//============================================================================================================================================
//============================================================================================================================================
OptimizedSimNLL::OptimizedSimNLL(const RooArgSet &obs, const RooArgSet &nuis, RooAbsPdf *pdf) :
    RooAbsReal(TString("nll_")+pdf->GetName(), ""),
    originalPdf_(pdf),
    obs_(obs),
    nuis_(nuis),
    params_("params","parameters",this),
    simCount_(0)
{
    init(obs_, nuis_, originalPdf_);
}

OptimizedSimNLL::OptimizedSimNLL(const OptimizedSimNLL &other, const char *name) :
    RooAbsReal(name ? name : (TString("nll_")+other.originalPdf_->GetName()).Data(), ""),
    originalPdf_(other.originalPdf_),
    obs_(other.obs_),
    nuis_(other.nuis_),
    params_("params",this,other.params_),
    simCount_(0)
{
    init(obs_, nuis_, originalPdf_);
}
void OptimizedSimNLL::init(const RooArgSet &obs, const RooArgSet &nuis, RooAbsPdf *pdf)
{
    RooAbsPdf *pdfclone = utils::fullClonePdf(pdf, pdfClonePieces_);
    std::auto_ptr<RooArgSet> params(pdfclone->getParameters(obs));
    params_.add(*params);

    RooArgList constraints;
    RooAbsPdf *factorPdf = utils::factorizePdf(obs, *pdfclone, constraints);
    // now prepare constraint terms
    int ic = 0;
    constraints_.resize(constraints.getSize());
    constraintNSets_.resize(constraints.getSize());
    DBG(DBG_SimNLL_ctor, (std::cout << "Found " << constraints_.size() << " constraints in " << pdf->GetName() << std::endl))
    std::auto_ptr<TIterator> iter(constraints.createIterator()); 
    std::auto_ptr<TIterator> iterNuis(nuis_.createIterator());
    for(RooAbsArg *arg = (RooAbsArg*) iter->Next(); arg != 0; arg = (RooAbsArg*) iter->Next(), ++ic) {
        // get nuisance pdf
        RooAbsPdf *pdf = dynamic_cast<RooAbsPdf *>(arg); assert(pdf != 0 && "Constraint is not a pdf");
        constraints_[ic] = pdf;
        // now we add all nuisance parameters that are parameters of this function
        std::auto_ptr<RooArgSet> params(pdf->getParameters(obs));
        RooStats::RemoveConstantParameters(&*params);
        iterNuis->Reset();
        for(RooAbsArg *npar = (RooAbsArg*) iterNuis->Next(); npar != 0; npar = (RooAbsArg*) iterNuis->Next()) {
            if (params->contains(*npar)) constraintNSets_[ic].add(*npar);
        }
        DBG(DBG_SimNLL_ctor, (std::cout << "  constraint " << pdf->GetName() << " (" << pdf->ClassName() << "). normset = "))
        DBG(DBG_SimNLL_ctor, (constraintNSets_[ic].Print("")))
    }
    // now look at top level pdf. if not already cloned, clone so we own it
    if (factorPdf == pdfclone) factorPdf = (RooAbsPdf*) pdfclone->clone(TString("clone_")+pdf->GetName());
    // now act differently if simpdf or not
    RooSimultaneous *simpdf = dynamic_cast<RooSimultaneous *>(factorPdf);
    if (simpdf) {
        simpdf_.reset(simpdf);
        std::auto_ptr<RooAbsCategoryLValue> catClone((RooAbsCategoryLValue*) simpdf->indexCat().Clone());
        nlls_.resize(catClone->numBins(NULL), 0);
        DBG(DBG_SimNLL_ctor, (std::cout << "Pdf " << pdf->GetName() <<" is a SimPdf over category " << catClone->GetName() << ", with " << nlls_.size() << " bins" << std::endl));
        for (int ib = 0, nb = nlls_.size(); ib < nb; ++ib) {
            catClone->setBin(ib);
            RooAbsPdf *pdf = simpdf->getPdf(catClone->getLabel());
            if (pdf != 0) {
                DBG(DBG_SimNLL_ctor, (std::cout << "   bin " << ib << " (label " << catClone->getLabel() << ") has pdf " << pdf->GetName() << " of type " << pdf->ClassName() << std::endl))
                nlls_[ib] = new OptimizedSimpleNLL(pdf);
                simCount_++;
            }
        }   
    } else {
        DBG(DBG_SimNLL_ctor, (std::cout << "Pdf " << pdf->GetName() <<" is a simple pdf."  << std::endl))
        nonsimpdf_.reset(factorPdf);
        nlls_.push_back(new OptimizedSimpleNLL(factorPdf));
        simCount_ = 1;
    }
}


OptimizedSimNLL::~OptimizedSimNLL() 
{
    for (std::vector<OptimizedSimpleNLL *>::iterator it = nlls_.begin(); it != nlls_.end(); ++it) {
        delete *it;
    }
    nlls_.clear();
}

void OptimizedSimNLL::setData(const RooAbsData &data) {
    if (simpdf_.get()) {
        DBG(DBG_SimNLL_setD, (std::cout << "Pdf is a simpdf. will have to split data " << std::endl))
        for (std::vector<OptimizedSimpleNLL *>::iterator it = nlls_.begin(); it != nlls_.end(); ++it) {
            if (*it != 0) { (*it)->setNoData(); }
        }
        int i, n = data.numEntries(); int currBin = -1, first = 0;
        RooAbsCategoryLValue *cat = 0;
        if (n) {
            DBGV(DBG_SimNLL_setD, utils::printRAD(&data))
            const RooArgSet *entry = data.get(0);
            cat = dynamic_cast<RooAbsCategoryLValue *>(entry->find(simpdf_->indexCat().GetName()));
            assert(cat != 0 && "Didn't find category in dataset");
        }
        for (i = 0; i < n; ++i) {
            data.get(i); int bin = cat->getBin();
            if (bin != currBin) { 
                if (currBin != -1) {
                    DBGV(DBG_SimNLL_setD, (std::cout << "  Range [" << first << ", " << i << ")  is in bin " << currBin << std::endl))
                    assert(currBin < int(nlls_.size()) && "Bin outside range");
                    if (nlls_[currBin] == 0) continue;
                    nlls_[currBin]->addData(data, first, i);
                }
                currBin = bin; first = i;
            }
        }
        if (n) {
            DBGV(DBG_SimNLL_setD, (std::cout << "  Range [" << first << ", " << n << ")  is in bin " << currBin << std::endl))
            assert(currBin < int(nlls_.size()) && "Bin outside range");
            nlls_[currBin]->addData(data, first, i);
        }
#if defined(DBG_SimNLL_setD) && (DBG_SimNLL_setD > 0)
        for (std::vector<OptimizedSimpleNLL *>::const_iterator it = nlls_.begin(); it != nlls_.end(); ++it) {
            if (*it == 0) continue;
            std::cout << " bin " << (it - nlls_.begin());
            if ((*it)->data() == 0) std::cout << " empty." << std::endl;
            else std::cout << " entries: " << (*it)->data()->numEntries() << std::endl;
        }
#endif
    } else {
        nlls_.front()->setData(data);
    }
    setValueDirty();
}

Double_t OptimizedSimNLL::evaluate() const {
#if defined(DBG_SimNLL_eval) && (DBG_SimNLL_eval > 1)
    std::cout << "At OptimizedSimNLL::evaluate(): " << std::endl;
    std::cout << "//-----------------------------------------------------------------------------\\\\ " << std::endl;
    std::auto_ptr<RooArgSet> dbgv(originalPdf_->getVariables()); dbgv->Print("V");
    std::cout << "\\\\-----------------------------------------------------------------------------// " << std::endl;
#endif
#if defined(DBG_SimNLL_cache) && (DBG_SimNLL_cache > 0)
    for (std::vector<OptimizedSimpleNLL *>::const_iterator it = nlls_.begin(), ed = nlls_.end(); it != ed; ++it) {
        std::cout << "   OptimizedSimpleNLL " << (*it)->GetName() << 
                     " value: " << ((*it)->isValueDirty() ? "dirty" : "clean") << 
                     ", mode: " << (*it)->operMode() <<
                     std::endl;
    }
#endif
    double ret = 0, logSimCount = (simCount_ > 1 ? log(double(simCount_)) : 0);
    for (std::vector<OptimizedSimpleNLL *>::const_iterator it = nlls_.begin(), ed = nlls_.end(); it != ed; ++it) {
        ret += (*it)->getVal(0);
        if (simCount_ > 1 && (*it)->data() != 0) ret += (*it)->data()->sumWeights()*logSimCount;
    }
    std::vector<RooAbsPdf *>::const_iterator itc = constraints_.begin(), edc = constraints_.end();
    std::vector<RooArgSet>::const_iterator itns = constraintNSets_.begin();
    double prod = 1.0;
    for (; itc != edc; ++itc, ++itns) {
        prod *= (*itc)->getVal(&*itns);
    }
    ret -= log(prod);
    DBG(DBG_SimNLL_eval, (std::cout << "OptimizedSimNLL::evaluate() = " << ret << std::endl))
#if defined(DBG_SimNLL_cache) && (DBG_SimNLL_cache > 0)
    for (std::vector<OptimizedSimpleNLL *>::const_iterator it = nlls_.begin(), ed = nlls_.end(); it != ed; ++it) {
        std::cout << "   OptimizedSimpleNLL " << (*it)->GetName() << " value: " << ((*it)->isValueDirty() ? "dirty" : "clean") << std::endl;
    }
#endif
    return ret;
}

RooArgSet* OptimizedSimNLL::getObservables(const RooArgSet* depList, Bool_t valueOnly) const 
{ 
    return new RooArgSet(); 
}
RooArgSet* OptimizedSimNLL::getParameters(const RooArgSet* depList, Bool_t stripDisconnected) const 
{ 
    RooArgSet deps(obs_); 
    if (depList) deps.add(*depList);
    return originalPdf_->getParameters(depList, stripDisconnected);
}

//============================================================================================================================================
//============================================================================================================================================
ProfiledLikelihoodRatioTestStatOpt::ProfiledLikelihoodRatioTestStatOpt(
    const RooArgSet & observables,
    RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, 
    const RooArgSet *nuisances, 
    const RooArgSet & paramsNull, const RooArgSet & paramsAlt)
:
    pdfNull_(&pdfNull), pdfAlt_(&pdfAlt),
    verbosity_(0)
{
    snapNull_.addClone(paramsNull);
    snapAlt_.addClone(paramsAlt);
    if (nuisances) nuisances_.addClone(*nuisances);
    nllNull_.reset(new OptimizedSimNLL(observables, nuisances ? *nuisances : RooArgSet(), pdfNull_));
    nllAlt_.reset( new OptimizedSimNLL(observables, nuisances ? *nuisances : RooArgSet(), pdfAlt_));
    paramsNull_.reset(nllNull_->getParameters((const RooArgSet *)0)); 
    paramsAlt_.reset( nllAlt_->getParameters((const RooArgSet *)0)); 
}

ProfiledLikelihoodRatioTestStatExt::ProfiledLikelihoodRatioTestStatExt(
    const RooArgSet & observables,
    RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt,
    const RooArgSet *nuisances,
    const RooArgSet & paramsNull, const RooArgSet & paramsAlt)
: 
    pdfNull_(&pdfNull), pdfAlt_(&pdfAlt),
    paramsNull_(pdfNull_->getVariables()), 
    paramsAlt_(pdfAlt_->getVariables()), 
    verbosity_(0)
{
    snapNull_.addClone(paramsNull);
    snapAlt_.addClone(paramsAlt);
    if (nuisances) nuisances_.addClone(*nuisances);
}



Double_t ProfiledLikelihoodRatioTestStatExt::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    *paramsNull_ = nuisances_;
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of null pdf " << pdfNull_->GetName()))
    DBGV(DBG_TestStat_params, (paramsNull_->Print("V")))
    double nullNLL = minNLL(*pdfNull_, data);

    *paramsAlt_ = nuisances_;
    *paramsAlt_ = snapAlt_;
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of alt pdf " << pdfAlt_->GetName()))
    DBGV(DBG_TestStat_params, (paramsAlt_->Print("V")))
    double altNLL = minNLL(*pdfAlt_, data);

    DBG(DBG_TestStat_params, (printf("Pln: null = %+8.4f, alt = %+8.4f\n", nullNLL, altNLL)))
    return nullNLL-altNLL;
}

Double_t ProfiledLikelihoodRatioTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    *paramsNull_ = nuisances_;
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of null pdf " << nllNull_->GetName()))
    DBGV(DBG_TestStat_params, (paramsNull_->Print("V")))
    double nullNLL = minNLL(*nllNull_, data);

    *paramsAlt_ = nuisances_;
    *paramsAlt_ = snapAlt_;
    DBGV(DBG_TestStat_params, (std::cout << "Parameters of alt pdf " << nllAlt_->GetName()))
    DBGV(DBG_TestStat_params, (paramsAlt_->Print("V")))
    double altNLL = minNLL(*nllAlt_, data);

    DBG(DBG_TestStat_params, (printf("Opt: null = %+8.4f, alt = %+8.4f\n", nullNLL, altNLL)))
    return nullNLL-altNLL;
}

double ProfiledLikelihoodRatioTestStatOpt::minNLL(OptimizedSimNLL &nll, RooAbsData &data) 
{
    if (verbosity_ > 0) std::cout << "Profiling likelihood." << nll.GetName() << std::endl;
    nll.setData(data);
#if defined(DBG_TestStat_NOFIT) && (DBG_TestStat_NOFIT > 0)
    return nll.getVal();
#endif
    RooMinimizer minim(nll);
    minim.setStrategy(0);
    minim.setPrintLevel(verbosity_-1);
    //minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
    minim.minimize("Minuit2","migrad");
    if (verbosity_ > 1) {
        std::auto_ptr<RooFitResult> res(minim.save());
        res->Print("V");
    }
    return nll.getVal();
}

double ProfiledLikelihoodRatioTestStatExt::minNLL(RooAbsPdf &pdf, RooAbsData &data) 
{
    if (verbosity_ > 0) std::cout << "Profiling likelihood for pdf " << pdf.GetName() << std::endl;
    std::auto_ptr<RooAbsReal> nll_(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
#if defined(DBG_TestStat_NOFIT) && (DBG_TestStat_NOFIT > 0)
    return nll_->getVal();
#endif
    RooMinimizer minim(*nll_);
    minim.setStrategy(0);
    minim.setPrintLevel(verbosity_-1);
    //minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
    minim.minimize("Minuit2","migrad");
    if (verbosity_ > 1) {
        std::auto_ptr<RooFitResult> res(minim.save());
        res->Print("V");
    }
    return nll_->getVal();
}

//============================================================================================================================================
//============================================================================================================================================

ProfiledLikelihoodTestStatOpt::ProfiledLikelihoodTestStatOpt(
    const RooArgSet & observables,
    RooAbsPdf &pdf, 
    const RooArgSet *nuisances, 
    const RooArgSet *globalObs, 
    const RooArgSet & params)
:
    pdf_(&pdf),
    verbosity_()
{
    params.snapshot(snap_,false);
    ((RooRealVar*)snap_.find(params.first()->GetName()))->setConstant(false);
    poi_.add(snap_);
    RooArgSet allpars; allpars.add(snap_);
    if (nuisances) { allpars.add(*nuisances, /*silent=*/true); nuisances_.add(*nuisances); snap_.addClone(*nuisances, /*silent=*/true); }
    if (globalObs) { globalObs_.add(*globalObs); }
    //utils::setAllConstant(nuisances_, false); 
    //nll_.reset(new OptimizedSimNLL(observables, allpars, pdf_));
    //params_.reset(nll_->getParameters((const RooArgSet *)0)); 
    params_.reset(pdf_->getParameters(observables));
    //utils::setAllConstant(nuisances_, true);
    //std::cout << "Observables: " << std::endl;
    //observables.Print("V");
    //std::cout << "All params: " << std::endl;
    //params_->Print("V");
    //std::cout << "Snapshot: " << std::endl;
    //snap_.Print("V");
    //std::cout << "POI: " << std::endl;
    //poi_.Print("V");
}


Double_t ProfiledLikelihoodTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    DBG(DBG_PLTestStat_pars, std::cout << "Being evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))
    // Initialize parameters
    *params_ = snap_;

    DBG(DBG_PLTestStat_pars, std::cout << "Being evaluated on " << data.GetName() << ": params after snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))
    // Initialize signal strength
    RooRealVar *rIn = (RooRealVar *) poi_.first();
    RooRealVar *r   = (RooRealVar *) params_->find(rIn->GetName());
    r->setMin(0); r->setMax(rIn->getVal());
    r->setConstant(false);
    DBG(DBG_PLTestStat_pars, (std::cout << "r In: "; rIn->Print("")))
    DBG(DBG_PLTestStat_pars, std::cout << "r before the fit: ") DBG(DBG_PLTestStat_pars, r->Print(""))

    // Perform unconstrained minimization (denominator)
    double nullNLL = minNLL(*pdf_, data);
    DBG(DBG_PLTestStat_pars, (std::cout << "r after the fit: "; r->Print("")))

    // Perform unconstrained minimization (numerator)
    r->setVal(rIn->getVal()); 
    r->setConstant(true);
    double thisNLL = minNLL(*pdf_, data);

    DBG(DBG_PLTestStat_pars, std::cout << "Was evaluated on " << data.GetName() << ": params before snapshot are " << std::endl)
    DBG(DBG_PLTestStat_pars, params_->Print("V"))

    // Reset nuisances, just in case
    *params_ = snap_;

    return thisNLL-nullNLL;
}

double ProfiledLikelihoodTestStatOpt::minNLL(OptimizedSimNLL &nll, RooAbsData &data) 
{
    //if (verbosity_ > -2) std::cout << "Profiling likelihood." << nll.GetName() << std::endl;
    nll.setData(data);
    RooMinimizer minim(nll);
    minim.setStrategy(0);
    minim.setPrintLevel(verbosity_-1);
    minim.minimize("Minuit2","migrad");
    if (verbosity_ > 1) {
        minim.hesse();
        std::auto_ptr<RooFitResult> res(minim.save());
        res->Print("V");
    }
    return nll.getVal();
    //return res->minNll();
}

double ProfiledLikelihoodTestStatOpt::minNLL(RooAbsPdf &pdf, RooAbsData &data) 
{
    std::auto_ptr<RooAbsReal> nll(pdf.createNLL(data, RooFit::Constrain(nuisances_)));
    RooMinimizer minim(*nll);
    minim.setStrategy(0);
    minim.setPrintLevel(verbosity_-1);
    minim.minimize("Minuit2","migrad");
    if (verbosity_ > 1) {
        minim.hesse();
        std::auto_ptr<RooFitResult> res(minim.save());
        res->Print("V");
    }
    return nll->getVal();
    //return res->minNll();
}


