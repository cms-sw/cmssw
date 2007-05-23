//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprRandomNumber.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprRandomNumber :
//         Generates an array of real random number from 0 to 1.
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
 
#ifndef _SprRandomNumber_HH
#define _SprRandomNumber_HH

#include "PhysicsTools/StatPatternRecognition/src/SprRanluxEngine.hh"

class SprRandomNumber
{
public:
  virtual ~SprRandomNumber() {}

  SprRandomNumber(int seed=0) : theRanluxEngine_(timeSeed(seed)) {}

  /*
    Initializes random number generator from seed.
    If negative, will generate seed from current time of day.
  */
  void init(int seed);

  /*
    Generate a sequence of random numbers between 0 and 1. 
    The array of doubles has to be allocated by the user before calling
    this routine. We have no way of checking array boundaries.
  */
  void sequence(double* seq, int npts);

private:
  static int timeSeed(int seed);

  SprRanluxEngine theRanluxEngine_;
};

#endif
