// File and Version Information:
//      $Id: SprTwoClassTaggerEff.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassTaggerEff :
//        Returns tagging efficiency.
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
 
#ifndef _SprTwoClassTaggerEff_HH
#define _SprTwoClassTaggerEff_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <iostream>


class SprTwoClassTaggerEff : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassTaggerEff() {}

  SprTwoClassTaggerEff() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    double mistag1 = 1;
    if( (wmis0+wcor1) > 0 )
      mistag1 = wmis0/(wmis0+wcor1);
    double mistag2 = 1;
    if( (wmis1+wcor0) > 0 )
      mistag2 = wmis1/(wmis1+wcor0);
    double q = 0;
    if( mistag1 < 0.5 )
      q += (wmis0+wcor1)*(1.-2.*mistag1)*(1.-2.*mistag1);
    if( mistag2 < 0.5 )
      q += (wmis1+wcor0)*(1.-2.*mistag2)*(1.-2.*mistag2);
    return q;
  }

  bool symmetric() const { return true; }

  double min() const { return 0; }
  double max() const { return SprUtils::max(); }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "SprTwoClassTaggerEff::dfom_dwmis0() not implemented." 
	      << std::endl;
    return 0;
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    std::cerr << "SprTwoClassTaggerEff::dfom_dwcor1() not implemented." 
	      << std::endl;
    return 0;
  }
};

#endif
