#include "HiggsAnalysis/CombinedLimit/interface/SimplerLikelihoodRatioTestStatExt.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"

SimplerLikelihoodRatioTestStatOpt::SimplerLikelihoodRatioTestStatOpt(
        const RooArgSet &obs, 
        RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, 
        const RooArgSet & paramsNull, const RooArgSet & paramsAlt, 
        bool factorize) :
    obs_(&obs)
{
    // take a snapshot of the pdf so we can modify it at will
    RooAbsPdf *cloneNull = utils::fullClonePdf(&pdfNull, pdfCompNull_);
    RooAbsPdf *cloneAlt  = utils::fullClonePdf(&pdfAlt, pdfCompAlt_);

    // factorize away constraints
    RooArgList constraints;
    pdfNull_ = factorize ? utils::factorizePdf(obs, *cloneNull, constraints) : cloneNull;
    pdfAlt_  = factorize ? utils::factorizePdf(obs, *cloneAlt,  constraints) : cloneAlt;
    if (pdfAlt_ == 0 || pdfNull_ == 0) throw std::invalid_argument("SimplerLikelihoodRatioTestStatOpt:: pdf does not depend on observables");
    if (pdfNull_ != cloneNull) pdfNullOwned_.reset(pdfNull_); // if new, then take
    if (pdfAlt_  != cloneAlt)  pdfAltOwned_.reset(pdfAlt_);   // ownership of it

    // take snapshot of parameters
    snapNull_.addClone(paramsNull);
    snapAlt_.addClone(paramsAlt);

    // if the pdf is a RooSimultaneous, unroll it
    simPdfNull_ =  dynamic_cast<RooSimultaneous *>(pdfNull_);
    simPdfAlt_ =  dynamic_cast<RooSimultaneous *>(pdfAlt_);
    if (simPdfNull_) unrollSimPdf(simPdfNull_, simPdfComponentsNull_);
    if (simPdfAlt_) unrollSimPdf(simPdfAlt_, simPdfComponentsAlt_);

    // optimize caching
    pdfNull_->optimizeCacheMode(*obs_);
    pdfAlt_->optimizeCacheMode(*obs_);

    // find nodes that directly depend on observables
    utils::getClients(*obs_, pdfCompNull_, pdfDepObs_);
    utils::getClients(*obs_, pdfCompAlt_,  pdfDepObs_);
}

SimplerLikelihoodRatioTestStatOpt::~SimplerLikelihoodRatioTestStatOpt() 
{
}

Double_t
SimplerLikelihoodRatioTestStatOpt::Evaluate(RooAbsData& data, RooArgSet& nullPOI) 
{
    // get parameters, if not there already
    if (paramsNull_.get() == 0) paramsNull_.reset(pdfNull_->getParameters(data));
    if (paramsAlt_.get() == 0)  paramsAlt_.reset(pdfAlt_->getParameters(data));

    // if the dataset is not empty, redirect pdf nodes to the dataset
    std::auto_ptr<TIterator> iterDepObs(pdfDepObs_.createIterator());
    bool nonEmpty = data.numEntries() > 0;
    if (nonEmpty) {
        const RooArgSet *entry = data.get(0);
        for (RooAbsArg *a = (RooAbsArg *) iterDepObs->Next(); a != 0; a = (RooAbsArg *) iterDepObs->Next()) {
            a->redirectServers(*entry);    
        }
    }

    // evaluate null pdf
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
    double nullNLL = simPdfNull_ ? evalSimNLL(data, simPdfNull_, simPdfComponentsNull_) : evalSimpleNLL(data, pdfNull_);

    // evaluate alt pdf
    *paramsAlt_ = snapAlt_;
    double altNLL = simPdfAlt_ ? evalSimNLL(data, simPdfAlt_, simPdfComponentsAlt_) : evalSimpleNLL(data, pdfAlt_);

    // put back links in pdf nodes, otherwise if the dataset goes out of scope they have dangling pointers
    if (nonEmpty) {
        iterDepObs->Reset();
        for (RooAbsArg *a = (RooAbsArg *) iterDepObs->Next(); a != 0; a = (RooAbsArg *) iterDepObs->Next()) {
            a->redirectServers(*obs_);    
        }
    }

    return nullNLL-altNLL;
}

void SimplerLikelihoodRatioTestStatOpt::unrollSimPdf(RooSimultaneous *simpdf, std::vector<RooAbsPdf *> &out) {
    // get a clone of the pdf category, so that I can use it to enumerate the pdf states
    std::auto_ptr<RooAbsCategoryLValue> catClone((RooAbsCategoryLValue*) simpdf->indexCat().Clone());
    out.resize(catClone->numBins(NULL), 0);
    //std::cout << "Pdf " << pdf->GetName() <<" is a SimPdf over category " << catClone->GetName() << ", with " << out.size() << " bins" << std::endl;

    // loop on the category state and fetch the pdfs
    for (int ib = 0, nb = out.size(); ib < nb; ++ib) {
        catClone->setBin(ib);
        RooAbsPdf *pdfi = simpdf->getPdf(catClone->getLabel());
        pdfi->optimizeCacheMode(*obs_);
        if (pdfi != 0) {
            //std::cout << "   bin " << ib << " (label " << catClone->getLabel() << ") has pdf " << pdfi->GetName() << " of type " << pdfi->ClassName() << std::endl;
            out[ib] = pdfi;
        }
    }
}

double SimplerLikelihoodRatioTestStatOpt::evalSimNLL(RooAbsData &data,  RooSimultaneous *pdf, std::vector<RooAbsPdf *> &components) {
    data.setDirtyProp(false);

    double sum = 0.0;
    int i, n = data.numEntries(); 

    // must fetch the category, from the dataset first, if it's not empty
    RooAbsCategoryLValue *cat = 0;
    if (n) {
        const RooArgSet *entry = data.get(0);
        cat = dynamic_cast<RooAbsCategoryLValue *>(entry->find(pdf->indexCat().GetName()));
        assert(cat != 0 && "Didn't find category in dataset");
    }

    // now loop on the dataset, and dispatch the request to the appropriate pdf
    std::vector<double> sumw(components.size(), 0);
    for (i = 0; i < n; ++i) {
        data.get(i); 
        double w = data.weight(); if (w == 0) continue;
        int bin = cat->getBin();
        assert(bin < int(components.size()) && "Bin outside range");
        if (components[bin] == 0) continue;
        sum  += -w*components[bin]->getLogVal(obs_);
        sumw[bin] +=  w;
    }

    // then compute extended term
    for (i = 0, n = components.size(); i < n; ++i) {
        if (components[i]) sum += components[i]->extendedTerm(UInt_t(sumw[i]), obs_);
    }
    return sum;
}

double SimplerLikelihoodRatioTestStatOpt::evalSimpleNLL(RooAbsData &data,  RooAbsPdf *pdf) {
    data.setDirtyProp(false);
    double sum = 0.0, sumw = 0.0;
    int i, n = data.numEntries(); 
    for (i = 0; i < n; ++i) {
        data.get(i); 
        double w = data.weight(); if (w == 0) continue;
        sum  += -w*pdf->getLogVal(obs_);
        sumw +=  w;
    }
    sum += pdf->extendedTerm(UInt_t(sumw), obs_);
    return sum;
}

// ===== This below is identical to the RooStats::SimpleLikelihoodRatioTestStat also in implementation
//       I've made a copy here just to be able to put some debug hooks inside.
#if 0

SimplerLikelihoodRatioTestStatExt::SimplerLikelihoodRatioTestStatExt(
        const RooArgSet &obs, 
        RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, 
        const RooArgSet & paramsNull, const RooArgSet & paramsAlt)
{
    RooArgList constraints;
    pdfNull_ = &pdfNull;
    pdfAlt_  = &pdfAlt;
    paramsNull_.reset(pdfNull_->getVariables());
    paramsAlt_.reset(pdfAlt_->getVariables());
    snapNull_.addClone(paramsNull);
    snapAlt_.addClone(paramsAlt);
}

SimplerLikelihoodRatioTestStatExt::~SimplerLikelihoodRatioTestStatExt() 
{
}

Double_t
SimplerLikelihoodRatioTestStatExt::Evaluate(RooAbsData& data, RooArgSet& nullPOI) 
{
    std::auto_ptr<RooAbsReal> nllNull_(pdfNull_->createNLL(data));
    std::auto_ptr<RooAbsReal> nllAlt_(pdfAlt_->createNLL(data));

    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
    double nullNLL = nllNull_->getVal();

    *paramsAlt_ = snapAlt_;
    double altNLL = nllAlt_->getVal();
    return nullNLL-altNLL;
}

#endif
