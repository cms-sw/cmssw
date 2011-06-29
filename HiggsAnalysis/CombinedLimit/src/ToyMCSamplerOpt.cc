#include "../interface/ToyMCSamplerOpt.h"
#include "../interface/utils.h"
#include <memory>
#include <RooSimultaneous.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooRandom.h>

ToyMCSamplerOpt::ToyMCSamplerOpt(RooStats::TestStatistic& ts, Int_t ntoys, RooAbsPdf *globalObsPdf) :
    ToyMCSampler(ts, ntoys),
    globalObsPdf_(globalObsPdf),
    globalObsValues_(0), globalObsIndex_(-1),
    weightVar_(0),
    _allVars(0)
{
}


ToyMCSamplerOpt::ToyMCSamplerOpt(const RooStats::ToyMCSampler &base) :
    ToyMCSampler(base),
    globalObsPdf_(0),
    globalObsValues_(0), globalObsIndex_(-1),
    weightVar_(0),
    _allVars(0)
{
}

ToyMCSamplerOpt::ToyMCSamplerOpt(const ToyMCSamplerOpt &other) :
    ToyMCSampler(other),
    globalObsPdf_(0),
    globalObsValues_(0), globalObsIndex_(-1),
    weightVar_(0),
    _allVars(0)
{
}

ToyMCSamplerOpt::~ToyMCSamplerOpt()
{
    delete weightVar_;
    for (std::map<RooAbsPdf *, toymcoptutils::SimPdfGenInfo *>::iterator it = genCache_.begin(), ed = genCache_.end(); it != ed; ++it) {
        delete it->second;
    }
    genCache_.clear();
    delete _allVars;
    delete globalObsValues_;
}


toymcoptutils::SinglePdfGenInfo::SinglePdfGenInfo(RooAbsPdf &pdf, RooArgSet& observables, bool preferBinned, const RooDataSet* protoData, int forceEvents) :
   mode_(preferBinned ? Binned : Unbinned),
   pdf_(&pdf),
   spec_(0) 
{
   if (pdf.getAttribute("forceGenBinned")) mode_ = Binned;
   else if (pdf.getAttribute("forceGenUnbinned")) mode_ = Unbinned;
   //else std::cout << "Pdf " << pdf.GetName() << " has no preference" << std::endl;

   RooArgSet *obs = pdf.getObservables(observables);
   observables_.add(*obs);
   delete obs;
#if ROOT_VERSION_CODE > ROOT_VERSION(5,29,99)
    #error no
   if (mode_ == Unbinned) spec_ = protoData ? pdf.prepareMultiGen(observables_, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true)) 
                                            : pdf.prepareMultiGen(observables_, RooFit::Extended());
#endif
}

toymcoptutils::SinglePdfGenInfo::~SinglePdfGenInfo()
{
    delete spec_;
}


RooAbsData *  
toymcoptutils::SinglePdfGenInfo::generate(const RooDataSet* protoData, int forceEvents) 
{
    assert(forceEvents == 0 && "SinglePdfGenInfo: forceEvents must be zero at least for now");
    RooAbsData *ret = 0;
    if (mode_ == Unbinned) {
#if ROOT_VERSION_CODE > ROOT_VERSION(5,29,99)
        #error no
        ret = pdf_->generate(*spec_);
#else
        ret = pdf_->generate(observables_, RooFit::Extended());
#endif
    } else {
#if ROOT_VERSION_CODE > ROOT_VERSION(5,29,99)
        #error no
        ret = protoData ? pdf_->generateBinned(observables_, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true))
                        : pdf_->generateBinned(observables_, RooFit::Extended());
#else
        // generateBinnedWorkaround
        RooDataSet *data =  pdf_->generate(observables_, RooFit::Extended());
        ret = new RooDataHist(data->GetName(), "", *data->get(), *data);
        delete data;
#endif
    }
    //std::cout << "Dataset generated from " << pdf_->GetName() << " (weighted? " << ret->isWeighted() << ")" << std::endl;
    //utils::printRAD(ret);
    return ret;
}

toymcoptutils::SimPdfGenInfo::SimPdfGenInfo(RooAbsPdf &pdf, RooArgSet& observables, bool preferBinned, const RooDataSet* protoData, int forceEvents) :
    cat_(0),
    observables_(observables)
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
        ret = new RooDataSet("gen", "", observables_, RooFit::Index((RooCategory&)*cat_), RooFit::Link(datasetPieces_) /*, RooFit::OwnLinked()*/);
        //ret = new RooDataSet("gen", "", observables_, RooFit::Index((RooCategory&)*cat_), RooFit::Import(datasetPieces_) /*, RooFit::OwnLinked()*/);
    } else ret = pdfs_[0]->generate(protoData, forceEvents);
    //std::cout << "Dataset generated from sim pdf (weighted? " << ret->isWeighted() << ")" << std::endl; utils::printRAD(ret);
    return ret;
}

void
ToyMCSamplerOpt::SetPdf(RooAbsPdf& pdf) 
{
    ToyMCSampler::SetPdf(pdf);
    delete _allVars; _allVars = 0; 
    delete globalObsValues_; globalObsValues_ = 0; globalObsIndex_ = -1;
}

#if ROOT_VERSION_CODE < ROOT_VERSION(5,29,0)
//--- Taken from SVN HEAD ------------
RooAbsData* ToyMCSamplerOpt::GenerateToyData(RooArgSet& /*nullPOI*/) const {
   // This method generates a toy data set for the given parameter point taking
   // global observables into account.

   if (fObservables == NULL) { 
      ooccoutE((TObject*)NULL,InputArguments) << "Observables not set." << endl; 
      return 0; 
   }

   RooArgSet observables(*fObservables);
   if(fGlobalObservables  &&  fGlobalObservables->getSize()) {
      observables.remove(*fGlobalObservables);

      
      // generate one set of global observables and assign it
      // has problem for sim pdfs
      RooSimultaneous* simPdf = dynamic_cast<RooSimultaneous*>(fPdf);
      if(globalObsPdf_ || !simPdf){
        if (globalObsValues_ == 0 || globalObsIndex_ == globalObsValues_->numEntries()) {
            delete globalObsValues_;
            globalObsValues_ = (globalObsPdf_ ? globalObsPdf_ : fPdf)->generate(*fGlobalObservables, fNToys);
            globalObsIndex_  = 0;
        }
	const RooArgSet *values = globalObsValues_->get(globalObsIndex_++);
        //std::cout << "Generated for " << fPdf->GetName() << std::endl; values->Print("V");
	if (!_allVars) {
	  _allVars = fPdf->getObservables(*fGlobalObservables);
	}
	*_allVars = *values;

      } else {
#if 0
	if (_pdfList.size()==0) {
	  TIterator* citer = simPdf->indexCat().typeIterator() ;
	  RooCatType* tt = NULL;
	  while((tt=(RooCatType*) citer->Next())) {
	    RooAbsPdf* pdftmp = simPdf->getPdf(tt->GetName()) ;
	    RooArgSet* globtmp = pdftmp->getObservables(*fGlobalObservables) ;
	    RooAbsPdf::GenSpec* gs = pdftmp->prepareMultiGen(*globtmp,RooFit::NumEvents(1)) ;
	    _pdfList.push_back(pdftmp) ;
	    _obsList.push_back(globtmp) ;
	    _gsList.push_back(gs) ;
	  }
	}

	list<RooArgSet*>::iterator oiter = _obsList.begin() ;
	list<RooAbsPdf::GenSpec*>::iterator giter = _gsList.begin() ;
	for (list<RooAbsPdf*>::iterator iter = _pdfList.begin() ; iter != _pdfList.end() ; ++iter, ++giter, ++oiter) {
	  //RooDataSet* tmp = (*iter)->generate(**oiter,1) ;	  
	  RooDataSet* tmp = (*iter)->generate(**giter) ;
	  **oiter = *tmp->get(0) ;
	  delete tmp ;	  
	}	
#else
        //try fix for sim pdf
        TIterator* iter = simPdf->indexCat().typeIterator() ;
        RooCatType* tt = NULL;
        while((tt=(RooCatType*) iter->Next())) {

            // Get pdf associated with state from simpdf
            RooAbsPdf* pdftmp = simPdf->getPdf(tt->GetName()) ;

            // Generate only global variables defined by the pdf associated with this state
            RooArgSet* globtmp = pdftmp->getObservables(*fGlobalObservables) ;
            RooDataSet* tmp = pdftmp->generate(*globtmp,1) ;

            // Transfer values to output placeholder
            *globtmp = *tmp->get(0) ;

            // Cleanup
            delete globtmp ;
            delete tmp ;
        }
#endif
      } 
}

   RooAbsData* data = NULL;

   if(!fImportanceDensity) {
      // no Importance Sampling
      data = Generate(*fPdf, observables);
   }else{

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

   return data;
}
#endif


RooAbsData *  
ToyMCSamplerOpt::Generate(RooAbsPdf& pdf, RooArgSet& observables, const RooDataSet* protoData, int forceEvents) const 
{
   if(fProtoData) {
      protoData = fProtoData;
      forceEvents = protoData->numEntries();
   }
   int events = forceEvents;
   if (events == 0) events = fNEvents;
   if (events != 0) return RooStats::ToyMCSampler::Generate(pdf, observables, protoData, forceEvents);
   toymcoptutils::SimPdfGenInfo *& info = genCache_[&pdf];
   if (info == 0) info = new toymcoptutils::SimPdfGenInfo(pdf, observables, fGenerateBinned, protoData, forceEvents);
   return info->generate(weightVar_, protoData, forceEvents);
}
