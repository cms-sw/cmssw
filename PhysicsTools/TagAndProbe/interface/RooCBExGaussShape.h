#ifndef ROO_CB_EX_GAUSS_SHAPE
#define ROO_CB_EX_GAUSS_SHAPE


#include "RooAbsPdf.h"
#include "RooAbsArg.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "TMath.h"
#include "Riostream.h"

class RooCBExGaussShape : public RooAbsPdf {
public:
  RooCBExGaussShape() {} ; 
  RooCBExGaussShape(const char *name, const char *title,
	      RooAbsReal& _m,
	      RooAbsReal& _m0,
	      RooAbsReal& _sigma,
	      RooAbsReal& _alpha,
	      RooAbsReal& _n,
              RooAbsReal& _sigma_2,
	      RooAbsReal& _frac
		    );

  RooCBExGaussShape(const RooCBExGaussShape& other, const char* name);
  inline virtual TObject* clone(const char* newname) const { return new RooCBExGaussShape(*this,newname);}
  inline ~RooCBExGaussShape(){}
  Double_t evaluate() const ;
  
  ClassDef(RooCBExGaussShape,1)

protected:

  RooRealProxy m ;
  RooRealProxy  m0 ;
  RooRealProxy  sigma ;
  RooRealProxy  alpha ;
  RooRealProxy  n ;
  RooRealProxy  sigma_2 ;
  RooRealProxy  frac ;

};

#endif
