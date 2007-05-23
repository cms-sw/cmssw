//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprLoss.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprLoss :
//         Collection of various per-event loss expressions.
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
 
#ifndef _SprLoss_HH
#define _SprLoss_HH

#include <cmath>

struct SprLoss
{
  static double quadratic(int y, double f) { return (f-y)*(f-y); }

  static double exponential(int y, double f) { 
    return exp(-f*(y==0 ? -1 : y)); 
  }

  static double correct_id(int y, double f) { return ( int(f)==y ? 0 : 1 ); }

  static double purity_ratio(int y, double f) { 
    return ( y==0 ? f/(1.-f) : (1.-f)/f );
  }
};

#endif
