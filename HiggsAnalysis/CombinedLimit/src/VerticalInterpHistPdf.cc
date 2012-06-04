#include "../interface/VerticalInterpHistPdf.h"

#include <cassert>
#include <memory>

#include "RooFit.h"
#include "Riostream.h"

#include "TIterator.h"
#include "RooRealVar.h"
#include "RooMsgService.h"



ClassImp(VerticalInterpHistPdf)


//_____________________________________________________________________________
VerticalInterpHistPdf::VerticalInterpHistPdf() :
   _cacheTotal(0),
   _cacheSingle(0),
   _cacheSingleGood(0) 
{
  // Default constructor
  _funcIter  = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
}


//_____________________________________________________________________________
VerticalInterpHistPdf::VerticalInterpHistPdf(const char *name, const char *title, const RooRealVar &x, const RooArgList& inFuncList, const RooArgList& inCoefList, Double_t smoothRegion, Int_t smoothAlgo) :
  RooAbsPdf(name,title),
  _x("x","Independent variable",this,const_cast<RooRealVar&>(x)),
  _funcList("funcList","List of pdfs",this),
  _coefList("coefList","List of coefficients",this), // we should get shapeDirty when coefficients change
  _smoothRegion(smoothRegion),
  _smoothAlgo(smoothAlgo),
  _cacheTotal(0),
  _cacheSingle(0),
  _cacheSingleGood(0) 
{ 

  if (inFuncList.getSize()!=2*inCoefList.getSize()+1) {
    coutE(InputArguments) << "VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Nfunc=1+2*Ncoef" << endl ;
    assert(0);
  }

  TIterator* funcIter = inFuncList.createIterator() ;
  RooAbsArg* func;
  while((func = (RooAbsArg*)funcIter->Next())) {
    RooAbsPdf *pdf = dynamic_cast<RooAbsPdf*>(func);
    if (!pdf) {
      coutE(InputArguments) << "ERROR: VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() << ") function  " << func->GetName() << " is not of type RooAbsPdf" << endl;
      assert(0);
    }
    RooArgSet *params = pdf->getParameters(RooArgSet(x));
    if (params->getSize() > 0) {
      coutE(InputArguments) << "ERROR: VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() << ") pdf  " << func->GetName() << " has some parameters." << endl;
      assert(0);
    }
    delete params;
    _funcList.add(*func) ;
  }
  delete funcIter;

  TIterator* coefIter = inCoefList.createIterator() ;
  RooAbsArg* coef;
  while((coef = (RooAbsArg*)coefIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "ERROR: VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal" << endl;
      assert(0);
    }
    _coefList.add(*coef) ;    
  }
  delete coefIter;

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

}




//_____________________________________________________________________________
VerticalInterpHistPdf::VerticalInterpHistPdf(const VerticalInterpHistPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _x("x",this,other._x),
  _funcList("funcList",this,other._funcList),
  _coefList("coefList",this,other._coefList),
  _smoothRegion(other._smoothRegion),
  _smoothAlgo(other._smoothAlgo),
  _cacheTotal(0),
  _cacheSingle(0),
  _cacheSingleGood(0) 
{
  // Copy constructor

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}



//_____________________________________________________________________________
VerticalInterpHistPdf::~VerticalInterpHistPdf()
{
  // Destructor
  delete _funcIter ;
  delete _coefIter ;
  if (_cacheTotal) {
      delete _cacheTotal;
      for (int i = 0; i < _funcList.getSize(); ++i) delete _cacheSingle[i]; 
      delete [] _cacheSingle;
      delete [] _cacheSingleGood;
  }
}

//_____________________________________________________________________________
Double_t VerticalInterpHistPdf::evaluate() const 
{
  if (_cacheTotal == 0) setupCaches();
#if 0
  printf("Evaluate called for x %f, cache status %d: ", _x.arg().getVal(), _sentry.good());
  int ndim = _coefList.getSize();
  for (int i = 0; i < ndim; ++i) { 
    RooAbsReal *rar = dynamic_cast<RooAbsReal *>(_coefList.at(i));
    printf("%s = %+6.4f  ", rar->GetName(), rar->getVal());
  }
  std::cout << std::endl;
#endif

  if (!_sentry.good()) syncTotal();
  int nbin = _cacheTotal->GetNbinsX(), ibin = _cacheTotal->FindBin(_x);
  if (ibin < 1) ibin = 1;
  else if (ibin > nbin) ibin = nbin;
  return _cacheTotal->GetBinContent(ibin);
}


void VerticalInterpHistPdf::syncComponent(int i) const {
    RooAbsPdf *pdfi = dynamic_cast<RooAbsPdf *>(_funcList.at(i));
    if (_cacheSingle[i] != 0) delete _cacheSingle[i];
    _cacheSingle[i] = pdfi->createHistogram("",dynamic_cast<const RooRealVar &>(_x.arg())); 
    _cacheSingle[i]->SetDirectory(0);
    if (_cacheSingle[i]->Integral("width")) { _cacheSingle[i]->Scale(1.0/_cacheSingle[i]->Integral("width")); }
    if (i > 0) {
        for (int b = 1, nb = _cacheSingle[i]->GetNbinsX(); b <= nb; ++b) {
            double y  = _cacheSingle[i]->GetBinContent(b);
            double y0 = _cacheSingle[0]->GetBinContent(b);
            if (_smoothAlgo < 0) {
                if (y > 0 && y0 > 0) {
                    double logk = log(y/y0);
                    // odd numbers correspond to up variations, even numbers to down variations,
                    // and down variations need -log(kappa) instead of log(kappa)
                    _cacheSingle[i]->SetBinContent(b, logk);
                } else {
                    _cacheSingle[i]->SetBinContent(b, 0);
                }
            } else {
                _cacheSingle[i]->SetBinContent(b, y - y0);
            }
        }
    }
    _cacheSingleGood[i] = true;
}

void VerticalInterpHistPdf::syncTotal() const {
    int ndim = _coefList.getSize();
    for (int i = 0; i < 2*ndim+1; ++i) {
        if (!_cacheSingleGood[i]) syncComponent(i);
    }
    for (int b = 1, nb = _cacheTotal->GetNbinsX(); b <= nb; ++b) {
        double val = _cacheSingle[0]->GetBinContent(b);
        _coefIter->Reset();
        for (int i = 0; i < ndim; ++i) {
            double dhi = _cacheSingle[2*i+1]->GetBinContent(b);
            double dlo = _cacheSingle[2*i+2]->GetBinContent(b);
            double x = (dynamic_cast<RooAbsReal *>(_coefIter->Next()))->getVal();
            double alpha = x * 0.5 * ((dhi-dlo) + (dhi+dlo)*smoothStepFunc(x));
            // alpha(0) = 0
            // alpha(+1) = dhi 
            // alpha(-1) = dlo
            // alpha(x >= +1) = |x|*dhi
            // alpha(x <= -1) = |x|*dlo
            // alpha is continuous and has continuous first and second derivative, as smoothStepFunc has them
            if (_smoothAlgo < 0) {
                val *= exp(alpha);
            } else {
                val += alpha; 
            }
        }    
        if (val <= 0) val = 1e-9;
        _cacheTotal->SetBinContent(b, val);
    }
    double norm = _cacheTotal->Integral("width");
    if (norm > 0) _cacheTotal->Scale(1.0/norm);
    _sentry.reset();
}

void VerticalInterpHistPdf::setupCaches() const {
    int ndim = _coefList.getSize();
    _cacheTotal     = (dynamic_cast<const RooRealVar &>(_x.arg())).createHistogram("total"); 
    _cacheTotal->SetDirectory(0);
    _cacheSingle     = new TH1*[2*ndim+1]; assert(_cacheSingle);
    _cacheSingleGood = new int[2*ndim+1];  assert(_cacheSingleGood);
    for (int i = 0; i < 2*ndim+1; ++i) { 
        _cacheSingle[i] = 0; 
        _cacheSingleGood[i] = 0; 
        syncComponent(i);  
    } 
    _sentry.addVars(_coefList); 
    syncTotal();
}

//=============================================================================================
ClassImp(FastVerticalInterpHistPdfBase)
ClassImp(FastVerticalInterpHistPdf)
ClassImp(FastVerticalInterpHistPdf2D)


//_____________________________________________________________________________
FastVerticalInterpHistPdfBase::FastVerticalInterpHistPdfBase() 
{
  // Default constructor
  _funcIter  = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
}


//_____________________________________________________________________________
FastVerticalInterpHistPdfBase::FastVerticalInterpHistPdfBase(const char *name, const char *title, const RooArgSet &obs, const RooArgList& inFuncList, const RooArgList& inCoefList, Double_t smoothRegion, Int_t smoothAlgo) :
  RooAbsPdf(name,title),
  _funcList("funcList","List of pdfs",this),
  _coefList("coefList","List of coefficients",this), // we should get shapeDirty when coefficients change
  _smoothRegion(smoothRegion),
  _smoothAlgo(smoothAlgo),
  _morphs(), _morphParams()
{ 

  if (inFuncList.getSize()!=2*inCoefList.getSize()+1) {
    coutE(InputArguments) << "VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Nfunc=1+2*Ncoef" << endl ;
    assert(0);
  }

  TIterator* funcIter = inFuncList.createIterator() ;
  RooAbsArg* func;
  while((func = (RooAbsArg*)funcIter->Next())) {
    RooAbsPdf *pdf = dynamic_cast<RooAbsPdf*>(func);
    if (!pdf) {
      coutE(InputArguments) << "ERROR: VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() << ") function  " << func->GetName() << " is not of type RooAbsPdf" << endl;
      assert(0);
    }
    RooArgSet *params = pdf->getParameters(obs);
    if (params->getSize() > 0) {
      coutE(InputArguments) << "ERROR: VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() << ") pdf  " << func->GetName() << " has some parameters." << endl;
      assert(0);
    }
    delete params;
    _funcList.add(*func) ;
  }
  delete funcIter;

  TIterator* coefIter = inCoefList.createIterator() ;
  RooAbsArg* coef;
  while((coef = (RooAbsArg*)coefIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "ERROR: VerticalInterpHistPdf::VerticalInterpHistPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal" << endl;
      assert(0);
    }
    _coefList.add(*coef) ;    
  }
  delete coefIter;

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

}

//_____________________________________________________________________________
FastVerticalInterpHistPdfBase::FastVerticalInterpHistPdfBase(const FastVerticalInterpHistPdfBase& other, const char* name) :
  RooAbsPdf(other,name),
  _funcList("funcList",this,other._funcList),
  _coefList("coefList",this,other._coefList),
  _smoothRegion(other._smoothRegion),
  _smoothAlgo(other._smoothAlgo),
  _morphs(), _morphParams()
{
  // Copy constructor
  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}


//_____________________________________________________________________________
FastVerticalInterpHistPdfBase::~FastVerticalInterpHistPdfBase()
{
  // Destructor
  delete _funcIter ;
  delete _coefIter ;
}


//_____________________________________________________________________________
Double_t FastVerticalInterpHistPdf::evaluate() const 
{
  if (_cache.size() == 0) setupCaches();

  if (!_sentry.good()) syncTotal();
  //return std::max<double>(1e-9, _cache.GetAt(_x));
  return _cache.GetAt(_x);
}

//_____________________________________________________________________________
Double_t FastVerticalInterpHistPdf2D::evaluate() const 
{
  if (_cache.size() == 0) setupCaches();

  if (!_sentry.good()) syncTotal();
  //return std::max<double>(1e-9, _cache.GetAt(_x, _y));
  return _cache.GetAt(_x, _y);
}



void FastVerticalInterpHistPdf::syncNominal() const {
    RooAbsPdf *pdf = dynamic_cast<RooAbsPdf *>(_funcList.at(0));
    const RooRealVar &x = dynamic_cast<const RooRealVar &>(_x.arg());
    std::auto_ptr<TH1> hist(pdf->createHistogram("",x));
    hist->SetDirectory(0); 
    _cacheNominal = FastHisto(*hist);
    _cacheNominal.Normalize();
    if (_smoothAlgo < 0) {
        _cacheNominalLog = _cacheNominal;
        _cacheNominalLog.Log();
    }
}

void FastVerticalInterpHistPdf2D::syncNominal() const {
    RooAbsPdf *pdf = dynamic_cast<RooAbsPdf *>(_funcList.at(0));
    const RooRealVar &x = dynamic_cast<const RooRealVar &>(_x.arg());
    const RooRealVar &y = dynamic_cast<const RooRealVar &>(_y.arg());
    const RooCmdArg &cond = _conditional ? RooFit::ConditionalObservables(RooArgSet(x)) : RooCmdArg::none();
    std::auto_ptr<TH1> hist(pdf->createHistogram("", x, RooFit::YVar(y), cond));
    hist->SetDirectory(0); 
    _cacheNominal = FastHisto2D(dynamic_cast<TH2F&>(*hist), _conditional);
    if (_conditional) _cacheNominal.NormalizeXSlices(); 
    else              _cacheNominal.Normalize(); 

    if (_smoothAlgo < 0) {
        _cacheNominalLog = _cacheNominal;
        _cacheNominalLog.Log();
    }
}



void FastVerticalInterpHistPdfBase::syncMorph(Morph &out, const FastTemplate &nominal, FastTemplate &lo, FastTemplate &hi) const {
    if (_smoothAlgo < 0)  {
        hi.LogRatio(nominal); lo.LogRatio(nominal);
        //printf("Log-ratios for dimension %d: \n", dim);  hi.Dump(); lo.Dump();
    } else {
        hi.Subtract(nominal); lo.Subtract(nominal);
        //printf("Differences for dimension %d: \n", dim);  hi.Dump(); lo.Dump();
    }
    FastTemplate::SumDiff(hi, lo, out.sum, out.diff);
    //printf("Sum and diff for dimension %d: \n", dim);  out.sum.Dump(); out.diff.Dump();
}

void FastVerticalInterpHistPdf::syncComponents(int dim) const {
    RooAbsPdf *pdfHi = dynamic_cast<RooAbsPdf *>(_funcList.at(2*dim+1));
    RooAbsPdf *pdfLo = dynamic_cast<RooAbsPdf *>(_funcList.at(2*dim+2));
    const RooRealVar &x = dynamic_cast<const RooRealVar &>(_x.arg());
    std::auto_ptr<TH1> histHi(pdfHi->createHistogram("",x)); histHi->SetDirectory(0); 
    std::auto_ptr<TH1> histLo(pdfLo->createHistogram("",x)); histLo->SetDirectory(0);

    FastHisto hi(*histHi), lo(*histLo); 
    //printf("Un-normalized templates for dimension %d: \n", dim);  hi.Dump(); lo.Dump();
    hi.Normalize(); lo.Normalize();
    //printf("Normalized templates for dimension %d: \n", dim);  hi.Dump(); lo.Dump();
    syncMorph(_morphs[dim], _cacheNominal, lo, hi);
}
void FastVerticalInterpHistPdf2D::syncComponents(int dim) const {
    RooAbsPdf *pdfHi = dynamic_cast<RooAbsPdf *>(_funcList.at(2*dim+1));
    RooAbsPdf *pdfLo = dynamic_cast<RooAbsPdf *>(_funcList.at(2*dim+2));
    const RooRealVar &x = dynamic_cast<const RooRealVar &>(_x.arg());
    const RooRealVar &y = dynamic_cast<const RooRealVar &>(_y.arg());
    const RooCmdArg &cond = _conditional ? RooFit::ConditionalObservables(RooArgSet(x)) : RooCmdArg::none();
    std::auto_ptr<TH1> histHi(pdfHi->createHistogram("", x, RooFit::YVar(y), cond)); histHi->SetDirectory(0); 
    std::auto_ptr<TH1> histLo(pdfLo->createHistogram("", x, RooFit::YVar(y), cond)); histLo->SetDirectory(0);

    FastHisto2D hi(dynamic_cast<TH2&>(*histHi), _conditional), lo(dynamic_cast<TH2&>(*histLo), _conditional); 
    //printf("Un-normalized templates for dimension %d: \n", dim);  hi.Dump(); lo.Dump();
    if (_conditional) {
        hi.NormalizeXSlices(); lo.NormalizeXSlices();
    } else {
        hi.Normalize(); lo.Normalize();
    }
    //printf("Normalized templates for dimension %d: \n", dim);  hi.Dump(); lo.Dump();
    syncMorph(_morphs[dim], _cacheNominal, lo, hi);
}


void FastVerticalInterpHistPdfBase::syncTotal(FastTemplate &cache, const FastTemplate &cacheNominal, const FastTemplate &cacheNominalLog) const {
    /* === how the algorithm works, in theory ===
     * let  dhi = h_hi - h_nominal
     *      dlo = h_lo - h_nominal
     * and x be the morphing parameter
     * we define alpha = x * 0.5 * ((dhi-dlo) + (dhi+dlo)*smoothStepFunc(x));
     * which satisfies:
     *     alpha(0) = 0
     *     alpha(+1) = dhi 
     *     alpha(-1) = dlo
     *     alpha(x >= +1) = |x|*dhi
     *     alpha(x <= -1) = |x|*dlo
     *     alpha is continuous and has continuous first and second derivative, as smoothStepFunc has them
     * === and in practice ===
     * we already have computed the histogram for diff=(dhi-dlo) and sum=(dhi+dlo)
     * so we just do template += (0.5 * x) * (diff + smoothStepFunc(x) * sum)
     * ========================================== */

    // start from nominal
    cache.CopyValues(_smoothAlgo < 0 ? cacheNominalLog : cacheNominal);
    //printf("Cache initialized to nominal template: \n");  cacheNominal.Dump();

    // apply all morphs one by one
    for (int i = 0, ndim = _coefList.getSize(); i < ndim; ++i) {
        double x = _morphParams[i]->getVal();
        double a = 0.5*x, b = smoothStepFunc(x);
        cache.Meld(_morphs[i].diff, _morphs[i].sum, a, b);    
        //printf("Merged transformation for dimension %d, x = %+5.3f, step = %.3f: \n", i, x, b);  cache.Dump();
    }

    // if necessary go back to linear scale
    if (_smoothAlgo < 0) {
        cache.Exp();
        //printf("Done exponential tranformation\n");  cache.Dump();
    } else {
        cache.CropUnderflows();
    }
    
    // mark as done
    _sentry.reset();
}

void FastVerticalInterpHistPdf::syncTotal() const {
    FastVerticalInterpHistPdfBase::syncTotal(_cache, _cacheNominal, _cacheNominalLog);

    // normalize the result
    _cache.Normalize(); 
    //printf("Normalized result\n");  _cache.Dump();
}

void FastVerticalInterpHistPdf2D::syncTotal() const {
    FastVerticalInterpHistPdfBase::syncTotal(_cache, _cacheNominal, _cacheNominalLog);
    // normalize the result
    if (_conditional) _cache.NormalizeXSlices(); 
    else              _cache.Normalize(); 
    //printf("Normalized result\n");  _cache.Dump();
}



void FastVerticalInterpHistPdf::setupCaches() const {
    int ndim = _coefList.getSize();

    _morphs.resize(ndim);
    _morphParams.resize(ndim);
    syncNominal();
    //printf("Nominal template has been set up: \n");  _cacheNominal.Dump();
    _coefIter->Reset();
    for (int i = 0; i < ndim; ++i) {
        _morphParams[i] = dynamic_cast<RooAbsReal *>(_coefIter->Next());
        _morphs[i].sum.Resize(_cacheNominal.size());
        _morphs[i].diff.Resize(_cacheNominal.size());
        syncComponents(i);
    } 
    _cache = FastHisto(_cacheNominal);

    _sentry.addVars(_coefList); 
    syncTotal();
}

void FastVerticalInterpHistPdf2D::setupCaches() const {
    int ndim = _coefList.getSize();

    _morphs.resize(ndim);
    _morphParams.resize(ndim);
    syncNominal();
    //printf("Nominal template has been set up: \n");  _cacheNominal.Dump();
    _coefIter->Reset();
    for (int i = 0; i < ndim; ++i) {
        _morphParams[i] = dynamic_cast<RooAbsReal *>(_coefIter->Next());
        _morphs[i].sum.Resize(_cacheNominal.size());
        _morphs[i].diff.Resize(_cacheNominal.size());
        syncComponents(i);
    } 
    _cache = FastHisto2D(_cacheNominal);

    _sentry.addVars(_coefList); 
    syncTotal();
}


