// File and Version Information:
//      $Id: SprTwoClassCrossEntropy.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassCrossEntropy :
//        Returns negative cross-entropy: p*log(p)+q*log(q)
//        where p and q are fractions of signal and background events
//        in the signal region, p+q=1
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
 
#ifndef _SprTwoClassCrossEntropy_HH
#define _SprTwoClassCrossEntropy_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>
#include <iostream>


class SprTwoClassCrossEntropy : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassCrossEntropy() {}

  SprTwoClassCrossEntropy() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    double wtot = wcor0+wmis0+wcor1+wmis1;
    if( wtot < SprUtils::eps() ) return this->min();
    double a(0), b(0);
    if( (wcor1+wmis0) > 0 )
      a -= (wcor1+wmis0)*log(wcor1+wmis0);
    if( wcor1 > 0 )
      a += wcor1*log(wcor1);
    if( wmis0 > 0 )
      a += wmis0*log(wmis0);
    if( (wcor0+wmis1) > 0 ) 
      b -= (wcor0+wmis1)*log(wcor0+wmis1);
    if( wcor0 > 0 )
      b += wcor0*log(wcor0);
    if( wmis1 > 0 )
      b += wmis1*log(wmis1);
    return (a+b)/wtot/log(2.);
  }

  bool symmetric() const { return true; }

  double min() const { return -1; }
  double max() const { return 0; }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivative for cross-entropy not implemented." << std::endl; 
    return 0;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivative for cross-entropy not implemented." << std::endl; 
    return 0;
  }
};

#endif
