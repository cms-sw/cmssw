//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprTransformation.hh,v 1.3 2006/11/26 02:04:30 narsky Exp $
//
// Description:
//      Class SprTransformation :
//         Collection of various 1D transformations.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprTransformation_HH
#define _SprTransformation_HH

#include <cmath>

struct SprTransformation
{
  static const double logitLow;
  static const double logitHigh;

  static double logit(double x) { 
    if( x < logitLow )  return 0.;
    if( x > logitHigh ) return 1.;
    return 1./(1.+exp(-x)); 
  }

  static double logitDouble(double x) { 
    if( x < 0.5*logitLow  ) return 0.;
    if( x > 0.5*logitHigh ) return 1.;
    return 1./(1.+exp(-2.*x)); 
  }

  static double logit_deriv(double x) {
    if( x < logitLow )  return 0.;
    if( x > logitHigh ) return 0.;
    return 1./(2.+exp(x)+exp(-x)); 
  }    

  static double logitInverse(double x) { return log(x/(1.-x)); }
  static double logitHalfInverse(double x) { return 0.5*logitInverse(x); }

  static double zeroOneToMinusPlusOne(double x) { return (2.*x-1.); }

  static double logitToMinusPlusOne(double x) { 
    return (2.*logit(x)-1.);
  }
  static double logitDoubleToMinusPlusOne(double x) { 
    return (2.*logitDouble(x)-1.);
  }
};

#endif
