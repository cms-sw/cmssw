/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooStepBernstein.h 28259 2009-04-16 16:21:16Z wouter $
 * Authors:                                                                  *
 *   Kyle Cranmer (L. Gray for step function piece)
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_STEPBERNSTEIN
#define ROO_STEPBERNSTEIN

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooStepBernstein : public RooAbsPdf {
public:

  RooStepBernstein() ;
  RooStepBernstein(const char *name, const char *title,
		   RooAbsReal& _x, RooAbsReal& _stepThresh,
		   const RooArgList& _coefList) ;

  RooStepBernstein(const RooStepBernstein& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooStepBernstein(*this, newname); }
  inline virtual ~RooStepBernstein() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

private:

  RooRealProxy _x;
  RooRealProxy _stepThresh;
  RooListProxy _coefList ;

  Double_t evaluate() const;

  ClassDef(RooStepBernstein,1) // Bernstein polynomial PDF with step function
};

#endif
