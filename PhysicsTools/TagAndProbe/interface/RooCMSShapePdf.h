/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOCMSSHAPEPDF
#define ROOCMSSHAPEPDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
 
class RooCMSShapePdf : public RooAbsPdf {
public:
  RooCMSShapePdf(const char *name, const char *title,
	      RooAbsReal& _x,
	      RooAbsReal& _alpha,
	      RooAbsReal& _beta,
	      RooAbsReal& _gamma,
	      RooAbsReal& _peak);
  RooCMSShapePdf(const RooCMSShapePdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooCMSShapePdf(*this,newname); }
  inline virtual ~RooCMSShapePdf() { }

protected:

  RooRealProxy x ;
  RooRealProxy alpha ;
  RooRealProxy beta ;
  RooRealProxy gamma ;
  RooRealProxy peak ;
  
  Double_t evaluate() const ;

private:

  //ClassDef(RooCMSShapePdf,0) // Your description goes here...
};
 
#endif

