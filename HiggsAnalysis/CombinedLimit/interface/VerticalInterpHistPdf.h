#ifndef ROO_VERTICA_INTERP_HIST
#define ROO_VERTICA_INTERP_HIST

/** Vertical interpolation between multiple histograms (or binned non-parametric pdfs, which eventually means histograms...)
    Does integration internally */

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "TH1.h"
#include "../interface/SimpleCacheSentry.h"
#include <cmath>

class VerticalInterpHistPdf : public RooAbsPdf {
public:

  VerticalInterpHistPdf() ;
  VerticalInterpHistPdf(const char *name, const char *title, const RooRealVar &x, const RooArgList& funcList, const RooArgList& coefList, Double_t smoothRegion=1., Int_t smoothAlgo=1) ;
  VerticalInterpHistPdf(const VerticalInterpHistPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new VerticalInterpHistPdf(*this,newname) ; }
  virtual ~VerticalInterpHistPdf() ;

  Bool_t selfNormalized() const { return kTRUE; }

  Double_t evaluate() const ;

  const RooArgList& funcList() const { return _funcList ; }
  const RooArgList& coefList() const { return _coefList ; }

protected:
  RooRealProxy   _x;
  RooListProxy _funcList ;   //  List of component FUNCs
  RooListProxy _coefList ;  //  List of coefficients
  Double_t     _smoothRegion;
  Int_t        _smoothAlgo;
  TIterator* _funcIter ;     //! Iterator over FUNC list
  TIterator* _coefIter ;    //! Iterator over coefficient list

  // TH1 containing the histogram of this pdf
  mutable SimpleCacheSentry _sentry; // !not to be serialized
  mutable TH1  *_cacheTotal;     //! not to be serialized
  // For additive morphing, histograms of fNominal, fUp and fDown
  // For multiplicative morphing, histograms of fNominal, log(fUp/fNominal), -log(fDown/fNominal);
  mutable TH1  **_cacheSingle; //! not to be serialized
  mutable int  *_cacheSingleGood; //! not to be serialized
private:

  ClassDef(VerticalInterpHistPdf,1) // 

  void setupCaches() const ;
  void syncTotal() const ;
  void syncComponent(int which) const ;
  // return a smooth function that is equal to +/-1 for |x| >= smoothRegion_ and it's null in zero
  inline double smoothStepFunc(double x) const { 
    if (fabs(x) >= _smoothRegion) return x > 0 ? +1 : -1;
    double xnorm = x/_smoothRegion, xnorm2 = xnorm*xnorm;
    return 0.125 * xnorm * (xnorm2 * (3.*xnorm - 10.) + 15);
  }
};

#endif
