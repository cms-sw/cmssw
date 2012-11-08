#include "../interface/ToyMCSamplerOpt.h"
#include "../interface/utils.h"
#include <memory>
#include <stdexcept>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <RooSimultaneous.h>
#include <RooRealVar.h>
#include <RooProdPdf.h>
#include <RooPoisson.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooRandom.h>
#include <../interface/ProfilingTools.h>

using namespace std;

ToyMCSamplerOpt::ToyMCSamplerOpt(RooStats::TestStatistic& ts, Int_t ntoys, RooAbsPdf *globalObsPdf, bool generateNuisances) :
    ToyMCSampler(ts, ntoys),
    globalObsPdf_(globalObsPdf),
    globalObsValues_(0), globalObsIndex_(-1),
    nuisValues_(0), nuisIndex_(-1),
    weightVar_(0)
{
    if (!generateNuisances) fPriorNuisance = 0; // set things straight from the beginning
}


ToyMCSamplerOpt::ToyMCSamplerOpt(const RooStats::ToyMCSampler &base) :
    ToyMCSampler(base),
    globalObsPdf_(0),
    globalObsValues_(0), globalObsIndex_(-1),
    weightVar_(0)
{
}

ToyMCSamplerOpt::ToyMCSamplerOpt(const ToyMCSamplerOpt &other) :
    ToyMCSampler(other),
    globalObsPdf_(0),
    globalObsValues_(0), globalObsIndex_(-1),
    weightVar_(0)
{
}

ToyMCSamplerOpt::~ToyMCSamplerOpt()
{
    delete weightVar_;
    for (std::map<RooAbsPdf *, toymcoptutils::SimPdfGenInfo *>::iterator it = genCache_.begin(), ed = genCache_.end(); it != ed; ++it) {
        delete it->second;
    }
    genCache_.clear();
    delete _allVars; _allVars = 0;
    delete globalObsValues_;
}


toymcoptutils::SinglePdfGenInfo::SinglePdfGenInfo(RooAbsPdf &pdf, const RooArgSet& observables, bool preferBinned, const RooDataSet* protoData, int forceEvents) :
   mode_(pdf.canBeExtended() ? (preferBinned ? Binned : Unbinned) : Counting),
   pdf_(&pdf),
   spec_(0),histoSpec_(0),keepHistoSpec_(0),weightVar_(0)
{
   if (pdf.canBeExtended()) {
       if (pdf.getAttribute("forceGenBinned")) mode_ = Binned;
       else if (pdf.getAttribute("forceGenPoisson")) mode_ = Poisson;
       else if (pdf.getAttribute("forceGenUnbinned")) mode_ = Unbinned;
   }

   RooArgSet *obs = pdf.getObservables(observables);
   observables_.add(*obs);
   delete obs;
   if (mode_ == Binned) {
      if (runtimedef::get("TMCSO_GenBinned")) mode_ = BinnedNoWorkaround;
      else if (runtimedef::get("TMCSO_GenBinnedWorkaround")) mode_ = Binned;
      else mode_ = Poisson;
   } else if (mode_ == Unbinned) {
       //if (!runtimedef::get("TMCSO_NoPrepareMultiGen")) {
       //    spec_ = protoData ? pdf.prepareMultiGen(observables_, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true)) 
       //                      : pdf.prepareMultiGen(observables_, RooFit::Extended());
       //}
   }
}

toymcoptutils::SinglePdfGenInfo::~SinglePdfGenInfo()
{
    delete spec_;
    delete weightVar_;
    delete histoSpec_;
}


RooAbsData *  
toymcoptutils::SinglePdfGenInfo::generate(const RooDataSet* protoData, int forceEvents) 
{
    assert(forceEvents == 0 && "SinglePdfGenInfo: forceEvents must be zero at least for now");
    RooAbsData *ret = 0;
    switch (mode_) {
        case Unbinned:
            if (spec_ == 0) spec_ = protoData ? pdf_->prepareMultiGen(observables_, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true))
                                              : pdf_->prepareMultiGen(observables_, RooFit::Extended());
            if (spec_) ret = pdf_->generate(*spec_);
            else ret = pdf_->generate(observables_, RooFit::Extended());
            break;
        case Binned:
            { // aka generateBinnedWorkaround
                RooDataSet *data =  pdf_->generate(observables_, RooFit::Extended());
                ret = new RooDataHist(data->GetName(), "", *data->get(), *data);
                delete data;
            }
            break;
        case BinnedNoWorkaround:
            ret = protoData ? pdf_->generateBinned(observables_, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true))
                            : pdf_->generateBinned(observables_, RooFit::Extended());
            break;
        case Poisson:
            ret = generateWithHisto(weightVar_, false);
            break;
        case Counting:
            ret = pdf_->generate(observables_, 1);
            break;
        default:
            throw std::logic_error("Mode not foreseen in SinglePdfGenInfo::generate");
    } 
    //std::cout << "Dataset generated from " << pdf_->GetName() << " (weighted? " << ret->isWeighted() << ")" << std::endl;
    //utils::printRAD(ret);
    return ret;
}

RooDataSet *  
toymcoptutils::SinglePdfGenInfo::generateAsimov(RooRealVar *&weightVar, double weightScale) 
{
    static int nPA = runtimedef::get("TMCSO_PseudoAsimov");
    if (nPA) return generatePseudoAsimov(weightVar, nPA, weightScale);
    return generateWithHisto(weightVar, true, weightScale);
}

RooDataSet *  
toymcoptutils::SinglePdfGenInfo::generatePseudoAsimov(RooRealVar *&weightVar, int nPoints, double weightScale) 
{
    if (mode_ == Unbinned) {
        double expEvents = pdf_->expectedEvents(observables_);
        std::auto_ptr<RooDataSet> data(pdf_->generate(observables_, nPoints));
        if (weightVar == 0) weightVar = new RooRealVar("_weight_","",1.0);
        RooArgSet obsPlusW(observables_); obsPlusW.add(*weightVar);
        RooDataSet *rds = new RooDataSet(data->GetName(), "", obsPlusW, weightVar->GetName());
        for (int i = 0; i < nPoints; ++i) {
            observables_ = *data->get(i);
            rds->add(observables_, weightScale*expEvents/nPoints);
        }
        return rds; 
    } else {
        return generateWithHisto(weightVar, true);
    }
}


RooDataSet *  
toymcoptutils::SinglePdfGenInfo::generateWithHisto(RooRealVar *&weightVar, bool asimov, double weightScale) 
{
    if (mode_ == Counting) return generateCountingAsimov();
    if (observables_.getSize() > 3) throw std::invalid_argument(std::string("ERROR in SinglePdfGenInfo::generateWithHisto for ") + pdf_->GetName() + ", more than 3 observable");
    RooArgList obs(observables_);
    RooRealVar *x = (RooRealVar*)obs.at(0);
    RooRealVar *y = obs.getSize() > 1 ? (RooRealVar*)obs.at(1) : 0;
    RooRealVar *z = obs.getSize() > 2 ? (RooRealVar*)obs.at(2) : 0;
    if (weightVar == 0) weightVar = new RooRealVar("_weight_","",1.0);

    RooCmdArg ay = (y ? RooFit::YVar(*y) : RooCmdArg::none());
    RooCmdArg az = (z ? RooFit::ZVar(*z) : RooCmdArg::none());

    if (histoSpec_ == 0) {
        histoSpec_ = pdf_->createHistogram("htemp", *x, ay, az); 
        histoSpec_->SetDirectory(0);
    } 

    double expectedEvents = pdf_->expectedEvents(observables_);
    histoSpec_->Scale(expectedEvents/ histoSpec_->Integral()); 
    RooArgSet obsPlusW(obs); obsPlusW.add(*weightVar);
    RooDataSet *data = new RooDataSet(TString::Format("%sData", pdf_->GetName()), "", obsPlusW, weightVar->GetName());
    RooAbsArg::setDirtyInhibit(true); // don't propagate dirty flags while filling histograms 
    switch (obs.getSize()) {
        case 1:
            for (int i = 1, n = histoSpec_->GetNbinsX(); i <= n; ++i) {
                x->setVal(histoSpec_->GetXaxis()->GetBinCenter(i));
                data->add(observables_,  weightScale*(asimov ? histoSpec_->GetBinContent(i) : RooRandom::randomGenerator()->Poisson(histoSpec_->GetBinContent(i))) );
            }
            break;
        case 2:
            {
            TH2& h2 = dynamic_cast<TH2&>(*histoSpec_);
            for (int ix = 1, nx = h2.GetNbinsX(); ix <= nx; ++ix) {
            for (int iy = 1, ny = h2.GetNbinsY(); iy <= ny; ++iy) {
                x->setVal(h2.GetXaxis()->GetBinCenter(ix));
                y->setVal(h2.GetYaxis()->GetBinCenter(iy));
                data->add(observables_, weightScale*(asimov ? h2.GetBinContent(ix,iy) : RooRandom::randomGenerator()->Poisson(h2.GetBinContent(ix,iy))) );
            } }
            }
            break;
        case 3:
            {
            TH3& h3 = dynamic_cast<TH3&>(*histoSpec_);
            for (int ix = 1, nx = h3.GetNbinsX(); ix <= nx; ++ix) {
            for (int iy = 1, ny = h3.GetNbinsY(); iy <= ny; ++iy) {
            for (int iz = 1, nz = h3.GetNbinsZ(); iz <= nz; ++iz) {
                x->setVal(h3.GetXaxis()->GetBinCenter(ix));
                y->setVal(h3.GetYaxis()->GetBinCenter(iy));
                z->setVal(h3.GetYaxis()->GetBinCenter(iz));
                data->add(observables_, weightScale*(asimov ? h3.GetBinContent(ix,iy,iz) : RooRandom::randomGenerator()->Poisson(h3.GetBinContent(ix,iy,iz))) );
            } } }
            }
    }
    RooAbsArg::setDirtyInhibit(false); // restore proper propagation of dirty flags
    if (!keepHistoSpec_) { delete histoSpec_; histoSpec_ = 0; }
    //std::cout << "Asimov dataset generated from " << pdf_->GetName() << " (sumw? " << data->sumEntries() << ", expected events " << expectedEvents << ")" << std::endl;
    //utils::printRDH(data);
    return data;
}


RooDataSet *  
toymcoptutils::SinglePdfGenInfo::generateCountingAsimov() 
{
    RooArgSet obs(observables_);
    RooProdPdf *prod = dynamic_cast<RooProdPdf *>(pdf_);
    RooPoisson *pois = 0;
    if (prod != 0) {
        setToExpected(*prod, observables_);
    } else if ((pois = dynamic_cast<RooPoisson *>(pdf_)) != 0) {
        setToExpected(*pois, observables_);
    } else throw std::logic_error("A counting model pdf must be either a RooProdPdf or a RooPoisson");
    RooDataSet *ret = new RooDataSet(TString::Format("%sData", pdf_->GetName()), "", obs);
    ret->add(obs);
    return ret;
}

void
toymcoptutils::SinglePdfGenInfo::setToExpected(RooProdPdf &prod, RooArgSet &obs) 
{
    std::auto_ptr<TIterator> iter(prod.pdfList().createIterator());
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        if (!a->dependsOn(obs)) continue;
        RooPoisson *pois = 0;
        if ((pois = dynamic_cast<RooPoisson *>(a)) != 0) {
            setToExpected(*pois, obs);
        } else {
            RooProdPdf *subprod = dynamic_cast<RooProdPdf *>(a);
            if (subprod) setToExpected(*subprod, obs);
            else throw std::logic_error("Illegal term in counting model: depends on observables, but not Poisson or Product");
        }
    }
}

void
toymcoptutils::SinglePdfGenInfo::setToExpected(RooPoisson &pois, RooArgSet &obs) 
{
    RooRealVar *myobs = 0;
    RooAbsReal *myexp = 0;
    std::auto_ptr<TIterator> iter(pois.serverIterator());
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        if (obs.contains(*a)) {
            assert(myobs == 0 && "SinglePdfGenInfo::setToExpected(RooPoisson): Two observables??");
            myobs = dynamic_cast<RooRealVar *>(a);
            assert(myobs != 0 && "SinglePdfGenInfo::setToExpected(RooPoisson): Observables is not a RooRealVar??");
        } else {
            assert(myexp == 0 && "SinglePdfGenInfo::setToExpected(RooPoisson): Two expecteds??");
            myexp = dynamic_cast<RooAbsReal *>(a);
            assert(myexp != 0 && "SinglePdfGenInfo::setToExpected(RooPoisson): Expectedis not a RooAbsReal??");
        }
    }
    assert(myobs != 0 && "SinglePdfGenInfo::setToExpected(RooPoisson): No observable?");
    assert(myexp != 0 && "SinglePdfGenInfo::setToExpected(RooPoisson): No expected?");
    myobs->setVal(myexp->getVal());
}

toymcoptutils::SimPdfGenInfo::SimPdfGenInfo(RooAbsPdf &pdf, const RooArgSet& observables, bool preferBinned, const RooDataSet* protoData, int forceEvents) :
    pdf_(&pdf),
    cat_(0),
    observables_(observables),
    copyData_(true)
{
    assert(forceEvents == 0 && "SimPdfGenInfo: forceEvents must be zero at least for now");
    RooSimultaneous *simPdf = dynamic_cast<RooSimultaneous *>(&pdf);
    if (simPdf) {
        cat_ = const_cast<RooAbsCategoryLValue *>(&simPdf->indexCat());
        int nbins = cat_->numBins((const char *)0);
        pdfs_.resize(nbins, 0);
        RooArgList dummy;
        for (int ic = 0; ic < nbins; ++ic) {
            cat_->setBin(ic);
            RooAbsPdf *pdfi = simPdf->getPdf(cat_->getLabel());
            if (pdfi == 0) throw std::logic_error(std::string("Unmapped category state: ") + cat_->getLabel());
            RooAbsPdf *newpdf = utils::factorizePdf(observables, *pdfi, dummy);
            pdfs_[ic] = new SinglePdfGenInfo(*newpdf, observables, preferBinned);
            if (newpdf != 0 && newpdf != pdfi) {
                ownedCrap_.addOwned(*newpdf); 
            }
        }
    } else {
        pdfs_.push_back(new SinglePdfGenInfo(pdf, observables, preferBinned, protoData, forceEvents));
    }
}

toymcoptutils::SimPdfGenInfo::~SimPdfGenInfo()
{
    for (std::vector<SinglePdfGenInfo *>::iterator it = pdfs_.begin(), ed = pdfs_.end(); it != ed; ++it) {
        delete *it;
    }
    pdfs_.clear();
    //for (std::map<std::string,RooDataSet*>::iterator it = datasetPieces_.begin(), ed = datasetPieces_.end(); it != ed; ++it) {
    for (std::map<std::string,RooAbsData*>::iterator it = datasetPieces_.begin(), ed = datasetPieces_.end(); it != ed; ++it) {
        delete it->second;
    }
    datasetPieces_.clear();
}


RooAbsData *  
toymcoptutils::SimPdfGenInfo::generate(RooRealVar *&weightVar, const RooDataSet* protoData, int forceEvents) 
{
    RooAbsData *ret = 0;
    TString retName =  TString::Format("%sData", pdf_->GetName());
    if (cat_ != 0) {
        //bool needsWeights = false;
        for (int i = 0, n = cat_->numBins((const char *)0); i < n; ++i) {
            if (pdfs_[i] == 0) continue;
            cat_->setBin(i);
            RooAbsData *&data =  datasetPieces_[cat_->getLabel()]; delete data;
            assert(protoData == 0);
            data = pdfs_[i]->generate(protoData); // I don't really know if protoData != 0 would make sense here
            if (data->isWeighted()) {
                if (weightVar == 0) weightVar = new RooRealVar("_weight_","",1.0);
                RooArgSet obs(*data->get()); 
                obs.add(*weightVar);
                RooDataSet *wdata = new RooDataSet(data->GetName(), "", obs, "_weight_");
                for (int i = 0, n = data->numEntries(); i < n; ++i) {
                    obs = *data->get(i);
                    if (data->weight()) wdata->add(obs, data->weight());
                }
                //std::cout << "DataHist was " << std::endl; utils::printRAD(data);
                delete data;
                data = wdata;
                //std::cout << "DataSet is " << std::endl; utils::printRAD(data);
            } 
            //if (data->isWeighted()) needsWeights = true;
        }
        if (copyData_) {
            //// slower but safer solution
            RooArgSet vars(observables_), varsPlusWeight(observables_); 
            if (weightVar) varsPlusWeight.add(*weightVar);
            ret = new RooDataSet(retName, "", varsPlusWeight, (weightVar ? weightVar->GetName() : 0));
            for (std::map<std::string,RooAbsData*>::iterator it = datasetPieces_.begin(), ed = datasetPieces_.end(); it != ed; ++it) {
                cat_->setLabel(it->first.c_str());
                for (unsigned int i = 0, n = it->second->numEntries(); i < n; ++i) {
                    vars = *it->second->get(i);
                    ret->add(vars, it->second->weight());
                }
            }
        } else {
            // not copyData is the "fast" mode used when generating toys as a ToyMCSampler.
            // this doesn't copy the data, so the toys cannot outlive this class and each new
            // toy over-writes the memory of the previous one.
            ret = new RooDataSet(retName, "", observables_, RooFit::Index((RooCategory&)*cat_), RooFit::Link(datasetPieces_) /*, RooFit::OwnLinked()*/);
        }
    } else ret = pdfs_[0]->generate(protoData, forceEvents);
    //std::cout << "Dataset generated from sim pdf (weighted? " << ret->isWeighted() << ")" << std::endl; utils::printRAD(ret);
    return ret;
}

RooAbsData *  
toymcoptutils::SimPdfGenInfo::generateAsimov(RooRealVar *&weightVar) 
{
    RooAbsData *ret = 0;
    TString retName =  TString::Format("%sData", pdf_->GetName());
    if (cat_ != 0) {
        //bool needsWeights = false;
        for (int i = 0, n = cat_->numBins((const char *)0); i < n; ++i) {
            if (pdfs_[i] == 0) continue;
            cat_->setBin(i);
            RooAbsData *&data =  datasetPieces_[cat_->getLabel()]; delete data;
            data = pdfs_[i]->generateAsimov(weightVar); 
        }
        if (copyData_) { 
            RooArgSet vars(observables_), varsPlusWeight(observables_); varsPlusWeight.add(*weightVar);
            ret = new RooDataSet(retName, "", varsPlusWeight, (weightVar ? weightVar->GetName() : 0));
            for (std::map<std::string,RooAbsData*>::iterator it = datasetPieces_.begin(), ed = datasetPieces_.end(); it != ed; ++it) {
                cat_->setLabel(it->first.c_str());
                for (unsigned int i = 0, n = it->second->numEntries(); i < n; ++i) {
                    vars = *it->second->get(i);
                    ret->add(vars, it->second->weight());
                }
            }
        } else {
            // not copyData is the "fast" mode used when generating toys as a ToyMCSampler.
            // this doesn't copy the data, so the toys cannot outlive this class and each new
            // toy over-writes the memory of the previous one.
            ret = new RooDataSet(retName, "", observables_, RooFit::Index((RooCategory&)*cat_), RooFit::Link(datasetPieces_) /*, RooFit::OwnLinked()*/);
        }
    } else ret = pdfs_[0]->generateAsimov(weightVar);
    //std::cout << "Asimov dataset generated from sim pdf " << pdf_->GetName() << " (sumw? " << ret->sumEntries() << ")" << std::endl; 
    //utils::printRAD(ret);
    return ret;
}

RooAbsData *  
toymcoptutils::SimPdfGenInfo::generateEpsilon(RooRealVar *&weightVar) 
{
    RooAbsData *ret = 0;
    TString retName =  TString::Format("%sData", pdf_->GetName());
    if (cat_ != 0) {
        //bool needsWeights = false;
        for (int i = 0, n = cat_->numBins((const char *)0); i < n; ++i) {
            if (pdfs_[i] == 0) continue;
            if (pdfs_[i]->mode() != SinglePdfGenInfo::Unbinned) continue;
            cat_->setBin(i);
            RooAbsData *&data =  datasetPieces_[cat_->getLabel()]; delete data;
            data = pdfs_[i]->generateAsimov(weightVar, 1e-9); 
        }
        if (copyData_) { 
            RooArgSet vars(observables_), varsPlusWeight(observables_); varsPlusWeight.add(*weightVar);
            ret = new RooDataSet(retName, "", varsPlusWeight, (weightVar ? weightVar->GetName() : 0));
            for (std::map<std::string,RooAbsData*>::iterator it = datasetPieces_.begin(), ed = datasetPieces_.end(); it != ed; ++it) {
                cat_->setLabel(it->first.c_str());
                for (unsigned int i = 0, n = it->second->numEntries(); i < n; ++i) {
                    vars = *it->second->get(i);
                    ret->add(vars, it->second->weight());
                }
            }
        } else {
            // not copyData is the "fast" mode used when generating toys as a ToyMCSampler.
            // this doesn't copy the data, so the toys cannot outlive this class and each new
            // toy over-writes the memory of the previous one.
            ret = new RooDataSet(retName, "", observables_, RooFit::Index((RooCategory&)*cat_), RooFit::Link(datasetPieces_) /*, RooFit::OwnLinked()*/);
        }
    } else ret = pdfs_[0]->generateAsimov(weightVar, 1e-9);
    //std::cout << "Asimov dataset generated from sim pdf " << pdf_->GetName() << " (sumw? " << ret->sumEntries() << ")" << std::endl; 
    //utils::printRAD(ret);
    return ret;
}


void
toymcoptutils::SimPdfGenInfo::setCacheTemplates(bool cache) 
{
    for (std::vector<SinglePdfGenInfo *>::const_iterator it = pdfs_.begin(), ed = pdfs_.end(); it != ed; ++it) {
        if (*it) (*it)->setCacheTemplates(cache);
    }
}

void
ToyMCSamplerOpt::SetPdf(RooAbsPdf& pdf) 
{
    ToyMCSampler::SetPdf(pdf);
    delete _allVars; _allVars = 0; 
    delete globalObsValues_; globalObsValues_ = 0; globalObsIndex_ = -1;
    delete nuisValues_; nuisValues_ = 0; nuisIndex_ = -1;
}

RooAbsData* ToyMCSamplerOpt::GenerateToyData(RooArgSet& /*nullPOI*/, double& weight) const {
   weight = 1;
   // This method generates a toy data set for the given parameter point taking
   // global observables into account.

   if (fObservables == NULL) { 
      ooccoutE((TObject*)NULL,InputArguments) << "Observables not set." << endl; 
      return 0; 
   }

   // generate nuisances
   RooArgSet saveNuis;
   if(fPriorNuisance && fNuisancePars && fNuisancePars->getSize() > 0) {
        if (nuisValues_ == 0 || nuisIndex_ == nuisValues_->numEntries()) {
            delete nuisValues_;
            nuisValues_ = fPriorNuisance->generate(*fNuisancePars, fNToys);
            nuisIndex_  = 0;
        }
        fNuisancePars->snapshot(saveNuis);
        const RooArgSet *values = nuisValues_->get(nuisIndex_++);
        RooArgSet pars(*fNuisancePars); pars = *values;
   }

   RooArgSet observables(*fObservables);
   if(fGlobalObservables  &&  fGlobalObservables->getSize()) {
      observables.remove(*fGlobalObservables);

      // generate one set of global observables and assign it
      assert(globalObsPdf_);
      if (globalObsValues_ == 0 || globalObsIndex_ == globalObsValues_->numEntries()) {
          delete globalObsValues_;
          globalObsValues_ = (globalObsPdf_ ? globalObsPdf_ : fPdf)->generate(*fGlobalObservables, fNToys);
          globalObsIndex_  = 0;
      }
      const RooArgSet *values = globalObsValues_->get(globalObsIndex_++);
      if (!_allVars) _allVars = fPdf->getObservables(*fGlobalObservables);
      *_allVars = *values;
   }

   RooAbsData* data = NULL;


#if ROOT_VERSION_CODE <  ROOT_VERSION(5,34,00)

   if(!fImportanceDensity) {
      // no Importance Sampling
      data = Generate(*fPdf, observables);
   }else{

      throw std::runtime_error("No importance sampling yet");

      // Importance Sampling
      RooArgSet* allVars = fPdf->getVariables();
      RooArgSet* allVars2 = fImportanceDensity->getVariables();
      allVars->add(*allVars2);
      const RooArgSet* saveVars = (const RooArgSet*)allVars->snapshot();

      // the number of events generated is either the given fNEvents or
      // in case this is not given, the expected number of events of
      // the pdf with a Poisson fluctuation
      int forceEvents = 0;
      if(fNEvents == 0) {
         forceEvents = (int)fPdf->expectedEvents(observables);
         forceEvents = RooRandom::randomGenerator()->Poisson(forceEvents);
      }

      // need to be careful here not to overwrite the current state of the
      // nuisance parameters, ie they must not be part of the snapshot
      if(fImportanceSnapshot) *allVars = *fImportanceSnapshot;

      // generate with the parameters configured in this class
      //   NULL => no protoData
      //   overwriteEvents => replaces fNEvents it would usually take
      data = Generate(*fImportanceDensity, observables, NULL, forceEvents);

      *allVars = *saveVars;
      delete allVars;
      delete allVars2;
      delete saveVars;
   }
#else
   // no Importance Sampling defined in ToyMCSampler of 5.34
   data = Generate(*fPdf, observables);
#endif   

   if (saveNuis.getSize()) { RooArgSet pars(*fNuisancePars); pars = saveNuis; }
   return data;
}


RooAbsData *  
ToyMCSamplerOpt::Generate(RooAbsPdf& pdf, RooArgSet& observables, const RooDataSet* protoData, int forceEvents) const 
{
   if(fProtoData) {
      protoData = fProtoData;
      forceEvents = protoData->numEntries();
   }
   int events = forceEvents;
   if (events == 0) events = fNEvents;
   if (events != 0) {
      assert(events == 1);
      assert(protoData == 0);
      RooAbsData *ret = pdf.generate(observables, events);
      return ret;
   }
   toymcoptutils::SimPdfGenInfo *& info = genCache_[&pdf];
   if (info == 0) { 
       info = new toymcoptutils::SimPdfGenInfo(pdf, observables, fGenerateBinned, protoData, forceEvents);
       info->setCopyData(false);
       if (!fPriorNuisance && importanceSnapshots_.empty()) info->setCacheTemplates(true);
   }
   return info->generate(weightVar_, protoData, forceEvents);
}
