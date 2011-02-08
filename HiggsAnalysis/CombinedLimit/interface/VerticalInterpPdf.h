#ifndef ROO_REAL_SUM_PDF_SAFE
#define ROO_REAL_SUM_PDF_SAFE

/** Vertical interpolation between multiple histograms (or pdfs).
    Based on RooRealSumPdf */

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"

class VerticalInterpPdf : public RooAbsPdf {
public:

  VerticalInterpPdf() ;
  VerticalInterpPdf(const char *name, const char *title, const RooArgList& funcList, const RooArgList& coefList, Double_t quadraticRegion=0., Int_t quadraticAlgo=0) ;
  VerticalInterpPdf(const VerticalInterpPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new VerticalInterpPdf(*this,newname) ; }
  virtual ~VerticalInterpPdf() ;

  Double_t evaluate() const ;
  virtual Bool_t checkObservables(const RooArgSet* nset) const ;	

  virtual Bool_t forceAnalyticalInt(const RooAbsArg&) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;

  const RooArgList& funcList() const { return _funcList ; }
  const RooArgList& coefList() const { return _coefList ; }

protected:
  
  class CacheElem : public RooAbsCacheElement {
  public:
    CacheElem()  {} ;
    virtual ~CacheElem() {} ; 
    virtual RooArgList containedArgs(Action) { RooArgList ret(_funcIntList) ; ret.add(_funcNormList) ; return ret ; }
    RooArgList _funcIntList ;
    RooArgList _funcNormList ;
  } ;
  mutable RooObjCacheManager _normIntMgr ; // The integration cache manager

  RooListProxy _funcList ;   //  List of component FUNCs
  RooListProxy _coefList ;  //  List of coefficients
  Double_t     _quadraticRegion;
  Int_t        _quadraticAlgo;
  TIterator* _funcIter ;     //! Iterator over FUNC list
  TIterator* _coefIter ;    //! Iterator over coefficient list

  Double_t interpolate(Double_t coeff, Double_t central, RooAbsReal *fUp, RooAbsReal *fDown) const ; 

private:

  ClassDef(VerticalInterpPdf,2) // PDF constructed from a sum of (non-pdf) functions
};

#endif
