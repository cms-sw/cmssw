/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOCBEXGAUSSSHAPE
#define ROOCBEXGAUSSSHAPE

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "TMath.h"
 
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
  RooCBExGaussShape(const RooCBExGaussShape& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooCBExGaussShape(*this,newname); }
  inline virtual ~RooCBExGaussShape() { }

protected:

  RooRealProxy m ;
  RooRealProxy  m0 ;
  RooRealProxy  sigma ;
  RooRealProxy  alpha ;
  RooRealProxy  n ;
  RooRealProxy  sigma_2 ;
  RooRealProxy  frac ;
  
  Double_t evaluate() const ;

private:

  ClassDef(RooCBExGaussShape,1) // Your description goes here...
};
 
#endif
