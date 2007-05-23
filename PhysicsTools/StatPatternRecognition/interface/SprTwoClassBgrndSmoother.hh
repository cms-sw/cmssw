// File and Version Information:
//      $Id: SprTwoClassBgrndSmoother.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
//
// Description:
//      Class SprTwoClassBgrndSmoother :
//         Reproduces Punzi's method implemented in SprTwoClassPunzi
//         but smoothes background using
//
//         B' = B + lambda*(1-B/omega)^2  for B<omega
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
 
#ifndef _SprTwoClassBgrndSmoother_HH
#define _SprTwoClassBgrndSmoother_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>
#include <iostream>
#include <cassert>


class SprTwoClassBgrndSmoother : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassBgrndSmoother() {}

  SprTwoClassBgrndSmoother(double bnorm, 
			   double lambda, 
			   double omega, 
			   double nSigmaCL=3.0) 
    : 
    SprAbsTwoClassCriterion(), 
    bnorm_(bnorm),
    lambda_(lambda),
    omega_(omega),
    nSigmaCL_(nSigmaCL) 
  {
    assert( bnorm_>0 );
    assert( lambda_>0 );
    assert( omega_>0 );
    assert( nSigmaCL_>=0 );
    std::cout << "Coefficients for BgrndSmoother are set to    " 
	      << "    Bnorm=" << bnorm_ 
	      << "    Lambda=" << lambda_ 
	      << "    Omega=" << omega_
	      << std::endl;
  }

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    if( wcor1 < SprUtils::eps() ) return 0;
    if(      wmis0 < SprUtils::eps() ) 
      wmis0  = lambda_;
    else if( wmis0 < omega_ )
      wmis0 += lambda_*pow((1.-wmis0/omega_),2);
    return wcor1/(0.5*nSigmaCL_ + sqrt(bnorm_*wmis0));
  }

  bool symmetric() const { return false; }

  double min() const { return 0; }
  double max() const { return SprUtils::max(); }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivatives not implemented for "
	      << "SprTwoClassBgrndSmoother." << std::endl;
    return 0;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivatives not implemented for "
	      << "SprTwoClassBgrndSmoother." << std::endl;
    return 0;
  }

private:
  double bnorm_;
  double lambda_;
  double omega_;
  double nSigmaCL_;
};

#endif
