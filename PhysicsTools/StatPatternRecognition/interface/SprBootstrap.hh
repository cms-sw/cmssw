//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprBootstrap.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprBootstrap :
//         Generates boostrap replicas.
/*
  In the context of HEP, Bootstrap can be used as a method for
  assessing estimator properties when inference is done on a small
  sample (without an option of getting more independent samples from
  the same distribution) and there is no good probability model for
  the underlying density. Bootstrap is described in plenty of books. A
  classic read is "An Introduction to Bootstrap" by Efron and Tibshirani.
*/
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
 
#ifndef _SprBootstrap_HH
#define _SprBootstrap_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprRandomNumber.hh"

class SprEmptyFilter;
class SprAbsFilter;


class SprBootstrap
{
public:
  virtual ~SprBootstrap() {}

  SprBootstrap(const SprAbsFilter* data, int seed=0) 
    : data_(data), generator_(seed) {}

  /*
    Initializes random number generator from seed.
  */
  void init(int seed) { generator_.init(seed); }

  /*
    Generates a bootstrap replica and returns ownership of replica to the user.
    User can specify how many points he wants to have in the replica.
    By default (npts=0) the size of the replica is equal to that of the 
    original sample.

    Plain replica is generated under assumption that all weights in the 
    original data sample are equal. Weighted replica takes proper account
    of event weights. plainReplica() is faster.
  */
  SprEmptyFilter* plainReplica(int npts=0);
  SprEmptyFilter* weightedReplica(int npts=0);

private:
  const SprAbsFilter* data_;
  SprRandomNumber generator_;
};

#endif
