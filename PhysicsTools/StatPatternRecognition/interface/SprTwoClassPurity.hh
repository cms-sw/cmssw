// File and Version Information:
//      $Id: SprTwoClassPurity.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassPurity :
//        Returns purity: S/(S+B)
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
 
#ifndef _SprTwoClassPurity_HH
#define _SprTwoClassPurity_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"


class SprTwoClassPurity : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassPurity() {}

  SprTwoClassPurity() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    if( (wcor1+wmis0) < SprUtils::eps() ) return 0;
    return wcor1/(wmis0+wcor1);
  }

  bool symmetric() const { return false; }

  double min() const { return 0; }
  double max() const { return 1; }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    if( (wcor1+wmis0) < SprUtils::eps() ) return 0;
    return -wcor1/pow(wmis0+wcor1,2);
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    if( (wcor1+wmis0) < SprUtils::eps() ) return 0;
    return wmis0/pow(wmis0+wcor1,2);
  }
};

#endif
