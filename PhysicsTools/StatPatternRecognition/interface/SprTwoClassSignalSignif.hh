// File and Version Information:
//      $Id: SprTwoClassSignalSignif.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassSignalSignif :
//        Returns signal significance: S/sqrt(S+B)
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
 
#ifndef _SprTwoClassSignalSignif_HH
#define _SprTwoClassSignalSignif_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>


class SprTwoClassSignalSignif : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassSignalSignif() {}

  SprTwoClassSignalSignif() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    if( wcor1 < SprUtils::eps() ) return 0;
    return wcor1/sqrt(wcor1+wmis0);
  }

  bool symmetric() const { return false; }

  double min() const { return 0; }
  double max() const { return SprUtils::max(); }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    if( wcor1 < SprUtils::eps() ) return 0;
    return -0.5*wcor1/pow(wcor1+wmis0,1.5);
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    if( wcor1 < SprUtils::eps() ) return 0;
    return (wcor1+2.*wmis0)/pow(wcor1+wmis0,1.5);
  }
};

#endif
