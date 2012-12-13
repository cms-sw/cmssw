#include "../interface/CachingNLL.h"
#include "../interface/utils.h"
#include <stdexcept>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooProduct.h>
#include "../interface/ProfilingTools.h"

//---- Uncomment this to get a '.' printed every some evals
//#define TRACE_NLL_EVALS

//---- Uncomment this and run with --perfCounters to get cache statistics
//#define DEBUG_CACHE

//---- Uncomment to enable Kahan's summation (if enabled at runtime with --X-rtd = ...
// http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//#define ADDNLL_KAHAN_SUM
#include "../interface/ProfilingTools.h"

//std::map<std::string,double> cacheutils::CachingAddNLL::offsets_;
bool cacheutils::CachingSimNLL::noDeepLEE_ = false;
bool cacheutils::CachingSimNLL::hasError_  = false;

//#define DEBUG_TRACE_POINTS
#ifdef DEBUG_TRACE_POINTS
namespace { 
    template<unsigned int>
    void tracePoint(const RooAbsCollection &point) {
        static const RooAbsCollection *lastPoint = 0;
        static std::vector<double> values;
        if (&point != lastPoint) {
            std::cout << "Arrived in a completely new point. " << std::endl;
            values.resize(point.getSize());
            RooLinkedListIter iter = point.iterator();
            for (RooAbsArg *a  = (RooAbsArg*)(iter.Next()); a != 0; a  = (RooAbsArg*)(iter.Next())) {
                RooRealVar *rrv = dynamic_cast<RooRealVar *>(a); if (!rrv) continue;
                values.push_back(rrv->getVal());
            }
            lastPoint = &point;
        } else {
            std::cout << "Moved: ";
            RooLinkedListIter iter = point.iterator();
            int i = 0;
            for (RooAbsArg *a  = (RooAbsArg*)(iter.Next()); a != 0; a  = (RooAbsArg*)(iter.Next())) {
                RooRealVar *rrv = dynamic_cast<RooRealVar *>(a); if (!rrv) continue;
                if (values[i] != rrv->getVal()) std::cout << a->GetName() << " " << values[i] << " => " << rrv->getVal() << "    "; 
                values[i++] = rrv->getVal();
            }
            std::cout << std::endl;
        }
    }
}
#define TRACE_POINT2(x,i)  ::tracePoint<i>(x);
#define TRACE_POINT(x)  ::tracePoint<0>(x);
#define TRACE_NLL(x)    std::cout << x << std::endl;
#else
#define TRACE_POINT2(x,i)
#define TRACE_POINT(x) 
#define TRACE_NLL(x) 
#endif

cacheutils::ArgSetChecker::ArgSetChecker(const RooAbsCollection &set) 
{
    std::auto_ptr<TIterator> iter(set.createIterator());
    for (RooAbsArg *a  = dynamic_cast<RooAbsArg *>(iter->Next()); 
                    a != 0; 
                    a  = dynamic_cast<RooAbsArg *>(iter->Next())) {
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv) { // && !rrv->isConstant()) { 
            vars_.push_back(rrv);
            vals_.push_back(rrv->getVal());
        }
    }
}

bool 
cacheutils::ArgSetChecker::changed(bool updateIfChanged) 
{
    std::vector<RooRealVar *>::const_iterator it = vars_.begin(), ed = vars_.end();
    std::vector<double>::iterator itv = vals_.begin();
    bool changed = false;
    for ( ; it != ed; ++it, ++itv) {
        double val = (*it)->getVal();
        if (val != *itv) { 
            //std::cerr << "var::CachingPdfable " << (*it)->GetName() << " changed: " << *itv << " -> " << val << std::endl;
            changed = true; 
            if (updateIfChanged) { *itv = val; }
            else break;
        }
    }
    return changed;
}

cacheutils::ValuesCache::ValuesCache(const RooAbsCollection &params, int size) :
    size_(1),
    maxSize_(size)
{
    assert(size <= MaxItems_);
    items[0] = new Item(params);
}
cacheutils::ValuesCache::ValuesCache(const RooAbsReal &pdf, const RooArgSet &obs, int size) :
    size_(1),
    maxSize_(size)
{
    assert(size <= MaxItems_);
    std::auto_ptr<RooArgSet> params(pdf.getParameters(obs));
    items[0] = new Item(*params);
}


cacheutils::ValuesCache::~ValuesCache() 
{
    for (int i = 0; i < size_; ++i) delete items[i];
}

void cacheutils::ValuesCache::clear() 
{
    for (int i = 0; i < size_; ++i) items[i]->good = false;
}

std::pair<std::vector<Double_t> *, bool> cacheutils::ValuesCache::get() 
{
    int found = -1; bool good = false;
    for (int i = 0; i < size_; ++i) {
        if (items[i]->good) {
            // valid entry, check if fresh
            if (!items[i]->checker.changed()) {
#ifdef DEBUG_CACHE
                PerfCounter::add(i == 0 ? "ValuesCache::get hit first" : "ValuesCache::get hit other");
#endif
                // fresh: done! 
                found = i; 
                good = true; 
                break;
            } 
        } else if (found == -1) {
            // invalid entry, can be replaced
            found = i;
#ifdef DEBUG_CACHE
            PerfCounter::add("ValuesCache::get hit invalid");
#endif
        }
    } 
    if (found == -1) {
        // all entries are valid but old 
#ifdef DEBUG_CACHE
        PerfCounter::add("ValuesCache::get miss");
#endif
        if (size_ < maxSize_) {
            // if I can, make a new entry
            items[size_] = new Item(items[0]->checker); // create a new item, copying the ArgSetChecker from the first one
            found = size_; 
            size_++;
        } else {
            // otherwise, pick the last one
            found = size_-1;
        }
    }
    // make sure new entry is the first one
    if (found != 0) {
        // remember what found is pointing to
        Item *f = items[found];
        // shift the other items down one place
        while (found > 0) { items[found] = items[found-1]; --found; } 
        // and put found on top
        items[found] = f;
    }
    if (!good) items[found]->checker.changed(true); // store new values in cache sentry
    items[found]->good = true;                      // mark this as valid entry
    return std::pair<std::vector<Double_t> *, bool>(&items[found]->values, good);
}

cacheutils::CachingPdf::CachingPdf(RooAbsReal *pdf, const RooArgSet *obs) :
    obs_(obs),
    pdfOriginal_(pdf),
    pdfPieces_(),
    pdf_(utils::fullCloneFunc(pdf, pdfPieces_)),
    lastData_(0),
    cache_(*pdf_,*obs_)
{
}

cacheutils::CachingPdf::CachingPdf(const CachingPdf &other) :
    obs_(other.obs_),
    pdfOriginal_(other.pdfOriginal_),
    pdfPieces_(),
    pdf_(utils::fullCloneFunc(pdfOriginal_, pdfPieces_)),
    lastData_(0),
    cache_(*pdf_,*obs_)
{
}

cacheutils::CachingPdf::~CachingPdf() 
{
}

const std::vector<Double_t> & 
cacheutils::CachingPdf::eval(const RooAbsData &data) 
{
#ifdef DEBUG_CACHE
    PerfCounter::add("CachingPdf::eval called");
#endif
    bool newdata = (lastData_ != &data);
    if (newdata) {
        lastData_ = &data;
        pdf_->optimizeCacheMode(*data.get());
        pdf_->attachDataSet(data);
        const_cast<RooAbsData*>(lastData_)->setDirtyProp(false);
        cache_.clear();
    }
    std::pair<std::vector<Double_t> *, bool> hit = cache_.get();
    if (!hit.second) {
        realFill_(data, *hit.first);
    } 
    return *hit.first;
}

void
cacheutils::CachingPdf::realFill_(const RooAbsData &data, std::vector<Double_t> &vals) 
{
#ifdef DEBUG_CACHE
    PerfCounter::add("CachingPdf::realFill_ called");
#endif
    int n = data.numEntries();
    vals.resize(n); // should be a no-op if size is already >= n.
    //std::auto_ptr<RooArgSet> params(pdf_->getObservables(*obs)); // for non-smart handling of pointers
    std::vector<Double_t>::iterator itv = vals.begin();
    for (int i = 0; i < n; ++i, ++itv) {
        data.get(i);
        //*params = *data.get(i); // for non-smart handling of pointers
        *itv = pdf_->getVal(obs_);
        //std::cout << " at i = " << i << " pdf = " << *itv << std::endl;
        TRACE_NLL("PDF value for " << pdf_->GetName() << " is " << *itv << " at this point.") 
        TRACE_POINT2(*obs_,1)
    }
}

cacheutils::CachingAddNLL::CachingAddNLL(const char *name, const char *title, RooAbsPdf *pdf, RooAbsData *data) :
    RooAbsReal(name, title),
    pdf_(pdf),
    params_("params","parameters",this),
    zeroPoint_(0)
{
    if (pdf == 0) throw std::invalid_argument(std::string("Pdf passed to ")+name+" is null");
    setData(*data);
    setup_();
}

cacheutils::CachingAddNLL::CachingAddNLL(const CachingAddNLL &other, const char *name) :
    RooAbsReal(name ? name : (TString("nll_")+other.pdf_->GetName()).Data(), ""),
    pdf_(other.pdf_),
    params_("params","parameters",this),
    zeroPoint_(0)
{
    setData(*other.data_);
    setup_();
}

cacheutils::CachingAddNLL::~CachingAddNLL() 
{
    for (int i = 0, n = integrals_.size(); i < n; ++i) delete integrals_[i];
    integrals_.clear();
}

cacheutils::CachingAddNLL *
cacheutils::CachingAddNLL::clone(const char *name) const 
{
    return new cacheutils::CachingAddNLL(*this, name);
}

void
cacheutils::CachingAddNLL::setup_() 
{
    const RooArgSet *obs = data_->get();
    for (int i = 0, n = integrals_.size(); i < n; ++i) delete integrals_[i];
    integrals_.clear();
    RooAddPdf *addpdf = 0;
    RooRealSumPdf *sumpdf = 0;
    if ((addpdf = dynamic_cast<RooAddPdf *>(pdf_)) != 0) {
        isRooRealSum_ = false;
        int npdf = addpdf->coefList().getSize();
        coeffs_.reserve(npdf);
        pdfs_.reserve(npdf);
        for (int i = 0; i < npdf; ++i) {
            RooAbsReal * coeff = dynamic_cast<RooAbsReal*>(addpdf->coefList().at(i));
            RooAbsPdf  * pdfi  = dynamic_cast<RooAbsPdf *>(addpdf->pdfList().at(i));
            coeffs_.push_back(coeff);
            pdfs_.push_back(CachingPdf(pdfi, obs));
        }
    } else if ((sumpdf = dynamic_cast<RooRealSumPdf *>(pdf_)) != 0) {
        isRooRealSum_ = true;
        int npdf = sumpdf->coefList().getSize();
        coeffs_.reserve(npdf);
        pdfs_.reserve(npdf);
        integrals_.reserve(npdf);
        for (int i = 0; i < npdf; ++i) {
            RooAbsReal * coeff = dynamic_cast<RooAbsReal*>(sumpdf->coefList().at(i));
            RooAbsReal * funci = dynamic_cast<RooAbsReal*>(sumpdf->funcList().at(i));
            /// Temporarily switch this off, it doesn't work. Don't know why, however.
            if (0 && typeid(*funci) == typeid(RooProduct)) {
                RooArgList obsDep, obsInd;
                obsInd.add(*coeff);
                utils::factorizeFunc(*obs, *funci, obsDep, obsInd);
                std::cout << "Entry " << i << ": coef name " << (coeff ? coeff->GetName()   : "null") << 
                                              "  type " << (coeff ? coeff->ClassName() :  "n/a") << std::endl;
                std::cout << "       " <<     "; func name " << (funci ? funci->GetName()   : "null") << 
                                              "  type " << (funci ? funci->ClassName() :  "n/a") << std::endl;
                std::cout << "Terms depending on observables: " << std::endl; obsDep.Print("V");
                std::cout << "Terms not depending on observables: " << std::endl; obsInd.Print("V");
                if (obsInd.getSize() > 1) {
                    coeff = new RooProduct(TString::Format("%s_x_%s_obsIndep", coeff->GetName(), funci->GetName()), "", RooArgSet(obsInd));
                    addOwnedComponents(RooArgSet(*coeff));
                }
                if (obsDep.getSize() > 1) {
                    funci = new RooProduct(TString::Format("%s_obsDep", funci->GetName()), "", RooArgSet(obsInd));
                    addOwnedComponents(RooArgSet(*funci));
                } else if (obsDep.getSize() == 1) {
                    funci = (RooAbsReal *) obsDep.first();
                } else throw std::logic_error("No part of pdf depends on observables?");
            }
            coeffs_.push_back(coeff);
            pdfs_.push_back(CachingPdf(funci, obs));
            integrals_.push_back(funci->createIntegral(*obs));
        }
    } else {
        std::string errmsg = "ERROR: CachingAddNLL: Pdf ";
        errmsg += pdf_->GetName();
        errmsg += " is neither a RooAddPdf nor a RooRealSumPdf, but a ";
        errmsg += pdf_->ClassName();
        throw std::invalid_argument(errmsg);
    }

    std::auto_ptr<RooArgSet> params(pdf_->getParameters(*data_));
    std::auto_ptr<TIterator> iter(params->createIterator());
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        //if (rrv != 0 && !rrv->isConstant()) params_.add(*rrv);
        if (rrv != 0) params_.add(*rrv);
    }
}

Double_t 
cacheutils::CachingAddNLL::evaluate() const 
{
#ifdef DEBUG_CACHE
    PerfCounter::add("CachingAddNLL::evaluate called");
#endif
    std::fill( partialSum_.begin(), partialSum_.end(), 0.0 );

    std::vector<RooAbsReal*>::iterator  itc = coeffs_.begin(), edc = coeffs_.end();
    std::vector<CachingPdf>::iterator   itp = pdfs_.begin();//,   edp = pdfs_.end();
    std::vector<Double_t>::const_iterator itw, bgw = weights_.begin();//,    edw = weights_.end();
    std::vector<Double_t>::iterator       its, bgs = partialSum_.begin(), eds = partialSum_.end();
    double sumCoeff = 0;
    //std::cout << "Performing evaluation of " << GetName() << std::endl;
    for ( ; itc != edc; ++itp, ++itc ) {
        // get coefficient
        Double_t coeff = (*itc)->getVal();
        if (isRooRealSum_) {
            sumCoeff += coeff * integrals_[itc - coeffs_.begin()]->getVal();
            //std::cout << "  coefficient = " << coeff << ", integral = " << integrals_[itc - coeffs_.begin()]->getVal() << std::endl;
        } else {
            sumCoeff += coeff;
        }
        // get vals
        const std::vector<Double_t> &pdfvals = itp->eval(*data_);
        // update running sum
        std::vector<Double_t>::const_iterator itv = pdfvals.begin();
        for (its = bgs; its != eds; ++its, ++itv) {
             *its += coeff * (*itv); // sum (n_i * pdf_i)
        }
    }
    // then get the final nll
    double ret = 0;
    bool fastExit = runtimedef::get("ADDNLL_FASTEXIT");
    for ( its = bgs, itw = bgw ; its != eds ; ++its, ++itw ) {
        if (*itw == 0) continue;
        if (!isnormal(*its) || *its <= 0) {
            std::cerr << "WARNING: underflow to " << *its << " in " << GetName() << std::endl; 
            if (!CachingSimNLL::noDeepLEE_) logEvalError("Number of events is negative or error"); else CachingSimNLL::hasError_ = true;
            if (fastExit) { ret += -9e9; break; }
        }
        double thispiece = (*itw) * (*its <= 0 ? -9e9 : log( ((*its) / sumCoeff) ));
        #ifdef ADDNLL_KAHAN_SUM
        static bool do_kahan = runtimedef::get("ADDNLL_KAHAN_SUM");
        if (do_kahan) {
            double kahan_y = thispiece  - compensation;
            double kahan_t = ret + kahan_y;
            double kahan_d = (kahan_t - ret);
            compensation = kahan_d - kahan_y;
            ret  = kahan_t;
        } else {
            ret += thispiece;
        }
        #else
        ret += thispiece;
        #endif
    }
    // then flip sign
    ret = -ret;
    // std::cout << "AddNLL for " << pdf_->GetName() << ": " << ret << std::endl;
    // and add extended term: expected - observed*log(expected);
    double expectedEvents = (isRooRealSum_ ? pdf_->getNorm(data_->get()) : sumCoeff);
    if (expectedEvents <= 0) {
        if (!CachingSimNLL::noDeepLEE_) logEvalError("Expected number of events is negative"); else CachingSimNLL::hasError_ = true;
        expectedEvents = 1e-6;
    }
    //ret += expectedEvents - UInt_t(sumWeights_) * log(expectedEvents); // no, doesn't work with Asimov dataset
    ret += expectedEvents - sumWeights_ * log(expectedEvents);
    ret += zeroPoint_;
    // std::cout << "     plus extended term: " << ret << std::endl;
    TRACE_NLL("AddNLL for " << pdf_->GetName() << ": " << ret)
    return ret;
}

void 
cacheutils::CachingAddNLL::setData(const RooAbsData &data) 
{
    //std::cout << "Setting data for pdf " << pdf_->GetName() << std::endl;
    //utils::printRAD(&data);
    data_ = &data;
    setValueDirty();
    sumWeights_ = 0.0;
    weights_.resize(data.numEntries());
    partialSum_.resize(data.numEntries());
    std::vector<Double_t>::iterator itw = weights_.begin();
    #ifdef ADDNLL_KAHAN_SUM
    double compensation = 0;
    #endif
    for (int i = 0, n = data.numEntries(); i < n; ++i, ++itw) {
        data.get(i);
        *itw = data.weight();
        #ifdef ADDNLL_KAHAN_SUM
        static bool do_kahan = runtimedef::get("ADDNLL_KAHAN_SUM");
        if (do_kahan) {
            double kahan_y = *itw - compensation;
            double kahan_t = sumWeights_ + kahan_y;
            double kahan_d = (kahan_t - sumWeights_);
            compensation = kahan_d - kahan_y;
            sumWeights_  = kahan_t;
        } else {
            sumWeights_ += *itw;
        }
        #else
        sumWeights_ += *itw;
        #endif
    }
    for (std::vector<CachingPdf>::iterator itp = pdfs_.begin(), edp = pdfs_.end(); itp != edp; ++itp) {
        itp->setDataDirty();
    }
}

RooArgSet* 
cacheutils::CachingAddNLL::getObservables(const RooArgSet* depList, Bool_t valueOnly) const 
{
    return new RooArgSet();
}

RooArgSet* 
cacheutils::CachingAddNLL::getParameters(const RooArgSet* depList, Bool_t stripDisconnected) const 
{
    return new RooArgSet(params_); 
}


cacheutils::CachingSimNLL::CachingSimNLL(RooSimultaneous *pdf, RooAbsData *data, const RooArgSet *nuis) :
    pdfOriginal_(pdf),
    dataOriginal_(data),
    nuis_(nuis),
    params_("params","parameters",this)
{
    setup_();
}

cacheutils::CachingSimNLL::CachingSimNLL(const CachingSimNLL &other, const char *name) :
    pdfOriginal_(other.pdfOriginal_),
    dataOriginal_(other.dataOriginal_),
    nuis_(other.nuis_),
    params_("params","parameters",this)
{
    setup_();
}

cacheutils::CachingSimNLL *
cacheutils::CachingSimNLL::clone(const char *name) const 
{
    return new cacheutils::CachingSimNLL(*this, name);
}

cacheutils::CachingSimNLL::~CachingSimNLL()
{
    for (std::vector<SimpleGaussianConstraint*>::iterator it = constrainPdfsFast_.begin(), ed = constrainPdfsFast_.end(); it != ed; ++it) {
        delete *it;
    }
}

void
cacheutils::CachingSimNLL::setup_() 
{
    // Allow runtime-flag to switch off logEvalErrors
    noDeepLEE_ = runtimedef::get("SIMNLL_NO_LEE");

    //RooAbsPdf *pdfclone = runtimedef::get("SIMNLL_CLONE") ? pdfOriginal_  : utils::fullClonePdf(pdfOriginal_, piecesForCloning_);
    RooAbsPdf *pdfclone = pdfOriginal_; // never clone

    //---- Instead of getting the parameters here, we get them from the individual constraint terms and single pdfs ----
    //---- This seems to save memory.
    //std::auto_ptr<RooArgSet> params(pdfclone->getParameters(*dataOriginal_));
    //params_.add(*params);

    RooArgList constraints;
    factorizedPdf_.reset(dynamic_cast<RooSimultaneous *>(utils::factorizePdf(*dataOriginal_->get(), *pdfclone, constraints)));
    
    RooSimultaneous *simpdf = factorizedPdf_.get();
    constrainPdfs_.clear(); 
    if (constraints.getSize()) {
        bool FastConstraints = runtimedef::get("SIMNLL_FASTGAUSS");
        //constrainPdfs_.push_back(new RooProdPdf("constraints","constraints", constraints));
        for (int i = 0, n = constraints.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = dynamic_cast<RooAbsPdf*>(constraints.at(i));
            if (FastConstraints && typeid(*pdfi) == typeid(RooGaussian)) {
                constrainPdfsFast_.push_back(new SimpleGaussianConstraint(dynamic_cast<const RooGaussian&>(*pdfi)));
                constrainZeroPointsFast_.push_back(0);
            } else {
                constrainPdfs_.push_back(pdfi);
                constrainZeroPoints_.push_back(0);
            }
            //std::cout << "Constraint pdf: " << constraints.at(i)->GetName() << std::endl;
            std::auto_ptr<RooArgSet> params(pdfi->getParameters(*dataOriginal_));
            params_.add(*params, false);
        }
    } else {
        std::cerr << "PDF didn't factorize!" << std::endl;
        std::cout << "Parameters: " << std::endl;
        std::auto_ptr<RooArgSet> params(pdfclone->getParameters(*dataOriginal_));
        params->Print("V");
        std::cout << "Obs: " << std::endl;
        dataOriginal_->get()->Print("V");
        factorizedPdf_.release();
        simpdf = dynamic_cast<RooSimultaneous *>(pdfclone);
    }

    
    std::auto_ptr<RooAbsCategoryLValue> catClone((RooAbsCategoryLValue*) simpdf->indexCat().Clone());
    pdfs_.resize(catClone->numBins(NULL), 0);
    //dataSets_.reset(dataOriginal_->split(pdfOriginal_->indexCat(), true));
    datasets_.resize(pdfs_.size(), 0);
    splitWithWeights(*dataOriginal_, simpdf->indexCat(), true);
    //std::cout << "Pdf " << simpdf->GetName() <<" is a SimPdf over category " << catClone->GetName() << ", with " << pdfs_.size() << " bins" << std::endl;
    for (int ib = 0, nb = pdfs_.size(); ib < nb; ++ib) {
        catClone->setBin(ib);
        RooAbsPdf *pdf = simpdf->getPdf(catClone->getLabel());
        if (pdf != 0) {
            RooAbsData *data = (RooAbsData *) datasets_[ib]; //dataSets_->FindObject(catClone->getLabel());
            //RooAbsData *data = (RooAbsData *) dataSets_->FindObject(catClone->getLabel());
            //std::cout << "   bin " << ib << " (label " << catClone->getLabel() << ") has pdf " << pdf->GetName() << " of type " << pdf->ClassName() << " and " << (data ? data->numEntries() : -1) << " dataset entries" << std::endl;
            if (data == 0) { throw std::logic_error("Error: no data"); }
            pdfs_[ib] = new CachingAddNLL(catClone->getLabel(), "", pdf, data);
            params_.add(pdfs_[ib]->params(), /*silent=*/true); 
        } else { 
            pdfs_[ib] = 0; 
            //std::cout << "   bin " << ib << " (label " << catClone->getLabel() << ") has no pdf" << std::endl;
        }
    }   

    setValueDirty();
}

Double_t 
cacheutils::CachingSimNLL::evaluate() const 
{
    TRACE_POINT(params_)
#ifdef DEBUG_CACHE
    PerfCounter::add("CachingSimNLL::evaluate called");
#endif
    double ret = 0;
    for (std::vector<CachingAddNLL*>::const_iterator it = pdfs_.begin(), ed = pdfs_.end(); it != ed; ++it) {
        if (*it != 0) {
            double nllval = (*it)->getVal();
            // what sanity check could I put here?
            ret += nllval;
        }
    }
    if (!constrainPdfs_.empty()) {
        /// ============= GENERIC CONSTRAINTS  =========
        std::vector<double>::const_iterator itz = constrainZeroPoints_.begin();
        for (std::vector<RooAbsPdf *>::const_iterator it = constrainPdfs_.begin(), ed = constrainPdfs_.end(); it != ed; ++it, ++itz) { 
            double pdfval = (*it)->getVal(nuis_);
            if (!isnormal(pdfval) || pdfval <= 0) {
                if (!noDeepLEE_) logEvalError((std::string("Constraint pdf ")+(*it)->GetName()+" evaluated to zero, negative or error").c_str());
                pdfval = 1e-9;
            }
            ret -= (log(pdfval) + *itz);
        }
        /// ============= FAST GAUSSIAN CONSTRAINTS  =========
        itz = constrainZeroPointsFast_.begin();
        for (std::vector<SimpleGaussianConstraint*>::const_iterator it = constrainPdfsFast_.begin(), ed = constrainPdfsFast_.end(); it != ed; ++it, ++itz) { 
            double logpdfval = (*it)->getLogValFast();
            ret -= (logpdfval + *itz);
        }
    }
#ifdef TRACE_NLL_EVALS
    static unsigned long _trace_ = 0; _trace_++;
    if (_trace_ % 10 == 0)  { putchar('.'); fflush(stdout); }
    //if (_trace_ % 250 == 0) { printf("               NLL % 10.4f after %10lu evals.\n", ret, _trace_); fflush(stdout); }
#endif
    TRACE_NLL("SimNLL for " << GetName() << ": " << ret)
    return ret;
}

void 
cacheutils::CachingSimNLL::setData(const RooAbsData &data) 
{
    dataOriginal_ = &data;
    //std::cout << "combined data has " << data.numEntries() << " dataset entries (sumw " << data.sumEntries() << ", weighted " << data.isWeighted() << ")" << std::endl;
    //utils::printRAD(&data);
    //dataSets_.reset(dataOriginal_->split(pdfOriginal_->indexCat(), true));
    splitWithWeights(*dataOriginal_, pdfOriginal_->indexCat(), true);
    for (int ib = 0, nb = pdfs_.size(); ib < nb; ++ib) {
        CachingAddNLL *canll = pdfs_[ib];
        if (canll == 0) continue;
        RooAbsData *data = datasets_[ib];
        //RooAbsData *data = (RooAbsData *) dataSets_->FindObject(canll->GetName());
        if (data == 0) { throw std::logic_error("Error: no data"); }
        //std::cout << "   bin " << ib << " (label " << canll->GetName() << ") has pdf " << canll->pdf()->GetName() << " of type " << canll->pdf()->ClassName() <<
        //             " and " << (data ? data->numEntries() : -1) << " dataset entries (sumw " << data->sumEntries() << ", weighted " << data->isWeighted() << ")" << std::endl;
        canll->setData(*data);
    }
}

void cacheutils::CachingSimNLL::splitWithWeights(const RooAbsData &data, const RooAbsCategory& splitCat, Bool_t createEmptyDataSets) {
    RooCategory *cat = dynamic_cast<RooCategory *>(data.get()->find(splitCat.GetName()));
    if (cat == 0) throw std::logic_error("Error: no category");
    int nb = cat->numBins((const char *)0), ne = data.numEntries();
    RooArgSet obs(*data.get()); obs.remove(*cat, true, true);
    RooRealVar weight("_weight_","",1);
    RooArgSet obsplus(obs); obsplus.add(weight);
    if (nb != int(datasets_.size())) throw std::logic_error("Number of categories changed"); // this can happen due to bugs in RooDataSet
    for (int ib = 0; ib < nb; ++ib) {
        if (datasets_[ib] == 0) datasets_[ib] = new RooDataSet("", "", obsplus, "_weight_");
        else datasets_[ib]->reset();
    }
    //utils::printRDH((RooAbsData*)&data);
    for (int i = 0; i < ne; ++i) {
        data.get(i); if (data.weight() == 0) continue;
        int ib = cat->getBin();
        //std::cout << "Event " << i << " of weight " << data.weight() << " is in bin " << ib << " label " << cat->getLabel() << std::endl;
        if (data.weight() > 0) datasets_[ib]->add(obs, data.weight());
    }
}

void cacheutils::CachingSimNLL::setZeroPoint() {
    for (std::vector<CachingAddNLL*>::const_iterator it = pdfs_.begin(), ed = pdfs_.end(); it != ed; ++it) {
        if (*it != 0) (*it)->setZeroPoint();
    }
    std::vector<double>::iterator itz = constrainZeroPoints_.begin();
    for (std::vector<RooAbsPdf *>::const_iterator it = constrainPdfs_.begin(), ed = constrainPdfs_.end(); it != ed; ++it, ++itz) {
        double pdfval = (*it)->getVal(nuis_);
        if (isnormal(pdfval) || pdfval > 0) *itz = -log(pdfval);
    }
    itz = constrainZeroPointsFast_.begin();
    for (std::vector<SimpleGaussianConstraint*>::const_iterator it = constrainPdfsFast_.begin(), ed = constrainPdfsFast_.end(); it != ed; ++it, ++itz) {
        double logpdfval = (*it)->getLogValFast();
        *itz = -logpdfval;
    }
    setValueDirty();
}

void cacheutils::CachingSimNLL::clearZeroPoint() {
    for (std::vector<CachingAddNLL*>::const_iterator it = pdfs_.begin(), ed = pdfs_.end(); it != ed; ++it) {
        if (*it != 0) (*it)->clearZeroPoint();
    }
    std::fill(constrainZeroPoints_.begin(), constrainZeroPoints_.end(), 0.0);
    std::fill(constrainZeroPointsFast_.begin(), constrainZeroPointsFast_.end(), 0.0);
    setValueDirty();
}

RooArgSet* 
cacheutils::CachingSimNLL::getObservables(const RooArgSet* depList, Bool_t valueOnly) const 
{
    return new RooArgSet();
}

RooArgSet* 
cacheutils::CachingSimNLL::getParameters(const RooArgSet* depList, Bool_t stripDisconnected) const 
{
    return new RooArgSet(params_); 
}
