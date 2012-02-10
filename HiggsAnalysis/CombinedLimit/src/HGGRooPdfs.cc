/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id: RooPower.cxx 25185 2008-08-20 14:00:42Z wouter $
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Power function p.d.f
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "../interface/HGGRooPdfs.h"
#include "RooRealVar.h"

ClassImp(RooPower)


//_____________________________________________________________________________
RooPower::RooPower(const char *name, const char *title,
			       RooAbsReal& _x, RooAbsReal& _c) :
  RooAbsPdf(name, title), 
  x("x","Dependent",this,_x),
  c("c","Power",this,_c)
{
}


//_____________________________________________________________________________
RooPower::RooPower(const RooPower& other, const char* name) :
  RooAbsPdf(other, name), x("x",this,other.x), c("c",this,other.c)
{
}


//_____________________________________________________________________________
Double_t RooPower::evaluate() const{
  //cout << "pow(x=" << x << ",c=" << c << ")=" << pow(x,c) << endl ;
  return pow(x,c);
}


//_____________________________________________________________________________
Int_t RooPower::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}


//_____________________________________________________________________________
Double_t RooPower::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  switch(code) {
  case 1: 
    {
      Double_t ret(0) ;
      if(c == 0.0) {
	ret = (x.max(rangeName) - x.min(rangeName));
      } else if (c== -1.0) {
	ret =  ( log( x.max(rangeName)) - log( x.min(rangeName)) );

      } else {
//	ret =  ( exp( c*x.max(rangeName) ) - exp( c*x.min(rangeName) ) )/c;
	ret =  ( pow( x.max(rangeName),  c+1 ) - pow( x.min(rangeName),c+1 ) )/(c+1);
      }

      //cout << "Int_exp_dx(c=" << c << ", xmin=" << x.min(rangeName) << ", xmax=" << x.max(rangeName) << ")=" << ret << endl ;
      return ret ;
    }
  }
  
  assert(0) ;
  return 0 ;
}
