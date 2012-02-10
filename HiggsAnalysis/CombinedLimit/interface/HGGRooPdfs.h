/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooPower.h,v 1.10 2007/07/12 20:30:49 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EXPONENTIAL
#define ROO_EXPONENTIAL

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooAbsReal;

class RooPower : public RooAbsPdf {
public:
  RooPower() {} ;
  RooPower(const char *name, const char *title,
		 RooAbsReal& _x, RooAbsReal& _c);
  RooPower(const RooPower& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooPower(*this,newname); }
  inline virtual ~RooPower() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

protected:
  RooRealProxy x;
  RooRealProxy c;

  Double_t evaluate() const;

private:
  ClassDef(RooPower,1) // Exponential PDF
};

#endif
