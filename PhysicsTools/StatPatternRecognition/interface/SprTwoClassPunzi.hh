// File and Version Information:
//      $Id: SprTwoClassPunzi.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
//
// Description:
//      Class SprTwoClassPunzi :
//        Returns the cross-section-independent FOM for the Poisson
//        distribution described by Punzi in Proceedings of Phystat 2003.
//        The 1st constructor argument is the relative data/MC normalization
//        to compute the background expected in the data.
//        The 2nd constructor argument is the number of Gaussian sigmas
//        that define both Type I and II errors. In this implementation,
//        Type I and II errors (alpha and beta in Punzi's notation)
//        are kept equal.
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
 
#ifndef _SprTwoClassPunzi_HH
#define _SprTwoClassPunzi_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>
#include <iostream>
#include <cassert>


class SprTwoClassPunzi : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassPunzi() {}

  SprTwoClassPunzi(double bnorm, double nSigmaCL=3.0) 
    : SprAbsTwoClassCriterion(), bnorm_(bnorm), nSigmaCL_(nSigmaCL) 
  {
    assert( bnorm_>0 );
    assert( nSigmaCL_>=0 );
    std::cout << "Background normalization for Punzi criterion set to " 
	      << bnorm_ << std::endl;
  }

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    if( wcor1 < SprUtils::eps() ) return 0;
    if( wmis0 < SprUtils::eps() ) return 2.*wcor1/nSigmaCL_;
    return wcor1/(0.5*nSigmaCL_ + sqrt(bnorm_*wmis0));
  }

  bool symmetric() const { return false; }

  double min() const { return 0; }
  double max() const { return SprUtils::max(); }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivatives not implemented for "
	      << "SprTwoClassPunzi." << std::endl;
    return 0;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "Derivatives not implemented for "
	      << "SprTwoClassPunzi." << std::endl;
    return 0;
  }

private:
  double bnorm_;
  double nSigmaCL_;
};

#endif
