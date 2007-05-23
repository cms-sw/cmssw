// File and Version Information:
//      $Id: SprTwoClassGiniIndex.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassGiniIndex :
//        Returns negative Gini index: -1+p^2+q^2
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
 
#ifndef _SprTwoClassGiniIndex_HH
#define _SprTwoClassGiniIndex_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>
#include <iostream>


class SprTwoClassGiniIndex : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassGiniIndex() {}

  SprTwoClassGiniIndex() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    double wtot = wcor0+wmis0+wcor1+wmis1;
    if( wtot < SprUtils::eps() ) return this->min();
    double a(0), b(0);
    if( (wcor1+wmis0) > 0 ) 
      a = -2.*wcor1*wmis0/(wmis0+wcor1);
    if( (wcor0+wmis1) > 0 ) 
      b = -2.*wcor0*wmis1/(wmis1+wcor0);
    return 2.*(a+b)/wtot;
  }

  bool symmetric() const { return true; }

  double min() const { return -1; }
  double max() const { return 0; }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivative for Gini index not implemented." << std::endl; 
    return 0;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivative for Gini index not implemented." << std::endl; 
    return 0;
  }

};

#endif
