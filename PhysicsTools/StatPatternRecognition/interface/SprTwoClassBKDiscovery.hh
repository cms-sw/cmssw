// File and Version Information:
//      $Id: SprTwoClassBKDiscovery.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprTwoClassBKDiscovery :
//        Returns discovery potential as per
//          Bityukov and Krasnikov hep-ph/0204326
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
 
#ifndef _SprTwoClassBKDiscovery_HH
#define _SprTwoClassBKDiscovery_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cmath>


class SprTwoClassBKDiscovery : public SprAbsTwoClassCriterion
{
public:
  virtual ~SprTwoClassBKDiscovery() {}

  SprTwoClassBKDiscovery() : SprAbsTwoClassCriterion() {}

  double fom(double wcor0, double wmis0, double wcor1, double wmis1) const {
    double a = 0;
    if( (wcor1+wmis0) > 0 ) a = sqrt(wcor1+wmis0);
    double b = 0;
    if( wmis0 > 0 ) b = sqrt(wmis0);
    return 2.*(a-b);
  }

  bool symmetric() const { return false; }

  double min() const { return 0; }
  double max() const { return SprUtils::max(); }

  double dfom_dwmis0(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    double a = 0;
    if( (wcor1+wmis0) > 0 ) a = 1./sqrt(wcor1+wmis0);
    double b = 0;
    if( wmis0 > 0 ) b = 1./sqrt(wmis0);
    return (a-b);
  }

  double dfom_dwcor1(double wcor0, double wmis0, 
		     double wcor1, double wmis1) const {
    if( (wcor1+wmis0) > 0 ) return 1./sqrt(wcor1+wmis0);
    return 0;
  }
};

#endif
