//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprIntegerPermutator.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprIntegerPermutator :
//         Generates permutations of N integer numbers from 1 to N.
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
 
#ifndef _SprIntegerPermutator_HH
#define _SprIntegerPermutator_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprRandomNumber.hh"

#include <vector>


class SprIntegerPermutator
{
public:
  virtual ~SprIntegerPermutator() {}

  SprIntegerPermutator(unsigned N, int seed=0);

  /*
    Initializes random number generator from seed.
    If negative, will generate seed from current time of day.
  */
  void init(int seed) { generator_.init(seed); }

  /*
    Generate a permutation of integer numbers from 0 to N.
  */
  bool sequence(std::vector<unsigned>& seq);

private:
  std::vector<unsigned> n_;
  SprRandomNumber generator_;
};

#endif
