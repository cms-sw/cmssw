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

#ifndef ROOPOWLAW
#define ROOPOWLAW

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "TMath.h"
 
class RooPowLaw : public RooAbsPdf {
public:
  RooPowLaw() {} ; 
  RooPowLaw(const char *name, const char *title,
	      RooAbsReal& _m,
	      RooAbsReal& _alpha
);
  RooPowLaw(const RooPowLaw& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooPowLaw(*this,newname); }
  inline virtual ~RooPowLaw() { }

protected:

  RooRealProxy m ;
  RooRealProxy  alpha ;
  
  Double_t evaluate() const ;

private:

  ClassDef(RooPowLaw,1) // Your description goes here...
};
 
#endif
