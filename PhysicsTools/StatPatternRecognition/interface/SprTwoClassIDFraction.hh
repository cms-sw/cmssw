// File and Version Information:
//      $Id: SprTwoClassIDFraction.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassIDFraction :
//        Returns correctly identified fraction of events.
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
 
#ifndef _SprTwoClassIDFraction_HH
#define _SprTwoClassIDFraction_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"


class SprTwoClassIDFraction : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassIDFraction() {}

  SprTwoClassIDFraction() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    double wtot = wcor0 + wmis0 + wcor1 + wmis1;
    if( wtot < SprUtils::eps() ) return 0;
    return (wcor0+wcor1)/wtot;
  }

  bool symmetric() const { return true; }

  double min() const { return 0.5; }
  double max() const { return 1; }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    double wtot = wcor0 + wmis0 + wcor1 + wmis1;
    if( wtot < SprUtils::eps() ) return 0;
    return -1./wtot;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    double wtot = wcor0 + wmis0 + wcor1 + wmis1;
    if( wtot < SprUtils::eps() ) return 0;
    return 1./wtot;
  }
};

#endif
