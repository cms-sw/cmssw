//$Id: SprRandomNumber.cc,v 1.3 2006/11/13 19:09:43 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRandomNumber.hh"

#include <sys/time.h>

using namespace std;

int SprRandomNumber::timeSeed(int seed)
{
  if( seed < 0 ) {
    struct timeval tp;
    gettimeofday(&tp, 0);
    return tp.tv_usec;
  }
  return seed;
}

void SprRandomNumber::init(int seed)
{
  theRanluxEngine_.setSeed(this->timeSeed(seed));
}

void SprRandomNumber::sequence(double* seq, int npts)
{
  // make array of uniform random numbers on [0,1]

  theRanluxEngine_.flatArray(npts, seq);
}
