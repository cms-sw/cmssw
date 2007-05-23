//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprIntegerBootstrap.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprIntegerBootstrap :
//         Generates a boostrap replica of a vector of integers from 0 to N
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
 
#ifndef _SprIntegerBootstrap_HH
#define _SprIntegerBootstrap_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprRandomNumber.hh"

#include <vector>
#include <set>
#include <cassert>

class SprEmptyFilter;
class SprAbsFilter;


class SprIntegerBootstrap
{
public:
  virtual ~SprIntegerBootstrap() {}

  SprIntegerBootstrap(unsigned dim, unsigned nsample, int seed=0) 
    : dim_(dim), nsample_(nsample), generator_(seed)
  {
    assert( dim_ > 0 );
    assert( nsample_ > 0 );
  }

  /*
    Initializes random number generator from seed.
    If negative, will generate seed from current time of day.
  */
  void init(int seed) { generator_.init(seed); }

  /*
    Generates a bootstrap replica. npts overrides nsample_
    The method with std::set returns distinct integers only stored in a set.
  */
  bool replica(std::vector<unsigned>& v, int npts=0);
  bool replica(std::set<unsigned>& v, int npts=0);

  /*
    Accessors.
  */
  unsigned dim() const { return dim_; }
  unsigned nsample() const { return nsample_; }

private:
  unsigned dim_;// range in which points will be generated: from 0 to dim_
  unsigned nsample_;// default number of points in the replica
  SprRandomNumber generator_;// random number generator
};

#endif
