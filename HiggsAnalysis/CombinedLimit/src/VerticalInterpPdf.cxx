#include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpPdf.h"

#include "RooFit.h"
#include "Riostream.h"

#include "TIterator.h"
#include "TList.h"
#include "RooRealProxy.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooRealConstant.h"
#include "RooRealIntegral.h"
#include "RooMsgService.h"



ClassImp(VerticalInterpPdf)


//_____________________________________________________________________________
VerticalInterpPdf::VerticalInterpPdf() 
{
  // Default constructor
  // coverity[UNINIT_CTOR]
  _funcIter  = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
  _quadraticRegion = 0;
}


//_____________________________________________________________________________
VerticalInterpPdf::VerticalInterpPdf(const char *name, const char *title, const RooArgList& inFuncList, const RooArgList& inCoefList, Double_t quadraticRegion) :
  RooAbsPdf(name,title),
  _normIntMgr(this,10),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _quadraticRegion(quadraticRegion)
{ 

  if (inFuncList.getSize()!=2*inCoefList.getSize()+1) {
    coutE(InputArguments) << "VerticalInterpPdf::VerticalInterpPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Nfunc=1+2*Ncoef" << endl ;
    assert(0);
  }

  TIterator* funcIter = inFuncList.createIterator() ;
  RooAbsArg* func;
  while((func = (RooAbsArg*)funcIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(func)) {
      coutE(InputArguments) << "ERROR: VerticalInterpPdf::VerticalInterpPdf(" << GetName() << ") function  " << func->GetName() << " is not of type RooAbsReal" << endl;
      assert(0);
    }
    _funcList.add(*func) ;
  }
  delete funcIter;

  TIterator* coefIter = inCoefList.createIterator() ;
  RooAbsArg* coef;
  while((coef = (RooAbsArg*)coefIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "ERROR: VerticalInterpPdf::VerticalInterpPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal" << endl;
      assert(0);
    }
    _coefList.add(*coef) ;    
  }
  delete coefIter;

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}




//_____________________________________________________________________________
VerticalInterpPdf::VerticalInterpPdf(const VerticalInterpPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _normIntMgr(other._normIntMgr,this),
  _funcList("!funcList",this,other._funcList),
  _coefList("!coefList",this,other._coefList)
{
  // Copy constructor

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}



//_____________________________________________________________________________
VerticalInterpPdf::~VerticalInterpPdf()
{
  // Destructor
  delete _funcIter ;
  delete _coefIter ;
}

//_____________________________________________________________________________
Double_t VerticalInterpPdf::evaluate() const 
{
  // Calculate the current value
  Double_t value(0) ;

  // Do running sum of coef/func pairs, calculate lastCoef.
  _funcIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* func = (RooAbsReal*)_funcIter->Next();

  Double_t central = func->getVal();
  value = central;

  while((coef=(RooAbsReal*)_coefIter->Next())) {
    Double_t coefVal = coef->getVal() ;
    RooAbsReal* funcUp = (RooAbsReal*)_funcIter->Next() ;
    RooAbsReal* funcDn = (RooAbsReal*)_funcIter->Next() ;
    value += interpolate(coefVal, central, funcUp, funcDn);
  }
  
  return value > 0 ? value : 1E-9 ;
}




//_____________________________________________________________________________
Bool_t VerticalInterpPdf::checkObservables(const RooArgSet* nset) const 
{
  // Check if FUNC is valid for given normalization set.
  // Coeffient and FUNC must be non-overlapping, but func-coefficient 
  // pairs may overlap each other
  //
  // In the present implementation, coefficients may not be observables or derive
  // from observables

  _coefIter->Reset() ;
  RooAbsReal* coef ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    if (coef->dependsOn(*nset)) {
      coutE(InputArguments) << "RooRealPdf::checkObservables(" << GetName() << "): ERROR coefficient " << coef->GetName() 
			    << " depends on one or more of the following observables" ; nset->Print("1") ;
      return true;
    }
  }

  _funcIter->Reset() ;
  RooAbsReal* func ;
  while((func = (RooAbsReal*)_funcIter->Next())) { 
    if (func->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "VerticalInterpPdf::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
			    << " and FUNC " << func->GetName() << " have one or more observables in common" << endl ;
      return true;
    }
  }
  
  return false;
}




//_____________________________________________________________________________
Int_t VerticalInterpPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					     const RooArgSet* normSet2, const char* /*rangeName*/) const 
{
  // Advertise that all integrals can be handled internally.

  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;

  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;
  RooArgSet* normSet = normSet2 ? getObservables(normSet2) : 0 ;


  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _normIntMgr.getObj(normSet,&analVars,&sterileIdx,0) ;
  if (cache) {
    return _normIntMgr.lastIndex()+1 ;
  }
  
  // Create new cache element
  cache = new CacheElem ;

  // Make list of function projection and normalization integrals 
  _funcIter->Reset() ;
  RooAbsReal *func ;
  while((func=(RooAbsReal*)_funcIter->Next())) {
    RooAbsReal* funcInt = func->createIntegral(analVars) ;
    cache->_funcIntList.addOwned(*funcInt) ;
    if (normSet && normSet->getSize()>0) {
      RooAbsReal* funcNorm = func->createIntegral(*normSet) ;
      cache->_funcNormList.addOwned(*funcNorm) ;
    }
  }

  // Store cache element
  Int_t code = _normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,0) ;

  if (normSet) {
    delete normSet ;
  }

  return code+1 ; 
}




//_____________________________________________________________________________
Double_t VerticalInterpPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet2, const char* /*rangeName*/) const 
{
  // Implement analytical integrations by deferring integration of component
  // functions to integrators of components

  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet2) ;

  RooAbsReal *coef;
  Double_t value = 0;

  // WVE needs adaptation for rangeName feature
  CacheElem* cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;

  TIterator* funcIntIter = cache->_funcIntList.createIterator() ;
  RooAbsReal *funcInt = (RooAbsReal *) funcIntIter->Next();
  Double_t central = funcInt->getVal();
  value += central;

  _coefIter->Reset() ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    Double_t coefVal = coef->getVal(normSet2) ;
    RooAbsReal * funcIntUp = (RooAbsReal*)funcIntIter->Next() ;
    RooAbsReal * funcIntDn = (RooAbsReal*)funcIntIter->Next() ;
    value += interpolate(coefVal, central, funcIntUp, funcIntDn);
  }
  
  delete funcIntIter ;
  
  Double_t normVal(1) ;
  if (normSet2) {
    normVal = 0 ;

    TIterator* funcNormIter = cache->_funcNormList.createIterator() ;

    RooAbsReal* funcNorm = (RooAbsReal*) funcNormIter->Next();
    central = funcNorm->getVal(normSet2) ;

    _coefIter->Reset() ;
    while((coef=(RooAbsReal*)_coefIter->Next())) {
      RooAbsReal *funcNormUp = (RooAbsReal*)funcNormIter->Next() ;
      RooAbsReal *funcNormDn = (RooAbsReal*)funcNormIter->Next() ;
      Double_t coefVal = coef->getVal(normSet2) ;
      normVal += interpolate(coefVal, central, funcNormUp, funcNormDn);
    }
    
    delete funcNormIter ;      
  }

  return ( value > 0 ? value : 1E-9 ) / normVal;
}

Double_t VerticalInterpPdf::interpolate(Double_t coeff, Double_t central, RooAbsReal *fUp, RooAbsReal *fDn) const  
{
    if (coeff == 0) return 0;
    if (fabs(coeff) >= _quadraticRegion) {
        return coeff * (coeff > 0 ? fUp->getVal() - central : central - fDn->getVal());
    } else {
        // quadratic interpolation coefficients between the three
        Double_t c_up  = coeff * (_quadraticRegion + coeff) / (2 * _quadraticRegion);
        Double_t c_dn  = coeff * (_quadraticRegion - coeff) / (2 * _quadraticRegion);
        Double_t c_cen = - coeff * fabs(coeff) / _quadraticRegion;
        return c_up * fUp->getVal() + c_dn * fDn->getVal() + c_cen * central;
    }

}
