//$Id: SprIntegerPermutator.cc,v 1.3 2006/11/13 19:09:42 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"

#include <algorithm>
#include <cassert>
#include <iostream>

using namespace std;


SprIntegerPermutator::SprIntegerPermutator(unsigned N, int seed) 
  : 
  n_(N), 
  generator_(seed) 
{
  for( unsigned i=0;i<N;i++ ) n_[i] = i;
}


bool SprIntegerPermutator::sequence(std::vector<unsigned>& seq)
{
  // init
  const unsigned N = n_.size();
  seq = n_;

  // generate random numbers
  double* r = new double [N];
  generator_.sequence(r,N);

  // make a permutation
  for( unsigned i=0;i<N;i++ ) {
    unsigned j = i + unsigned((N-i)*r[i]);
    assert( j>=i && j<N );
    swap(seq[i],seq[j]);
  }

  // cleanup
  delete [] r;

  // exit
  return true;
}
