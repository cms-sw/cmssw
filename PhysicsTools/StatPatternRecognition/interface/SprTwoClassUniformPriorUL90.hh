// File and Version Information:
//      $Id: SprTwoClassUniformPriorUL90.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassUniformPriorUL90 :
//        Returns the inverse of an approximate 90% upper limit 
//        computed with a Bayesian formula using the uniform prior.
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
 
#ifndef _SprTwoClassUniformPriorUL90_HH
#define _SprTwoClassUniformPriorUL90_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>
#include <iostream>


class SprTwoClassUniformPriorUL90 : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassUniformPriorUL90() {}

  SprTwoClassUniformPriorUL90() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    if( wcor1 < SprUtils::eps() ) return 0;
    if( wmis0 < SprUtils::eps() ) return wcor1/(2.303+1.313*wcor1);
    double a0 = 2.303 + 1.153*pow(wmis0,0.605);
    double a1 = 1.313*exp(-0.0873*pow(wmis0,0.470));
    double invUL = wcor1 / (a0+a1*wcor1);
    return invUL;
  }

  bool symmetric() const { return false; }

  double min() const { return 0; }
  double max() const { return SprUtils::max(); }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivatives not implemented for "
	      << "SprTwoClassUniformPriorUL90." << std::endl;
    return 0;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivatives not implemented for "
	      << "SprTwoClassUniformPriorUL90." << std::endl;
    return 0;
  }
};

#endif
