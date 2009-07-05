//$Id: SprBootstrap.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

using namespace std;


SprEmptyFilter* SprBootstrap::plainReplica(int npts)
{
  // init
  int size = data_->size();
  if( size == 0 ) return 0;
  if( npts<=0 || npts>size ) npts = size;
  SprData* replica = data_->emptyCopy();
  vector<double> weights;

  // make array of uniform random numbers on [0,1]
  double* r = new double [npts];
  generator_.sequence(r,npts);
  int iuse = -1;
  for( int i=0;i<npts;i++ ) {
    iuse = int(r[i] * size);
    if( iuse>=0 && iuse<size ) {
      replica->uncheckedInsert((*data_)[iuse]);
      weights.push_back(data_->w(iuse));
    }
  }
  delete [] r;

  // sanity check
  if( static_cast<int>(replica->size())!=npts || static_cast<int>(weights.size())!=npts ) {
    delete replica;
    return 0;
  }

  // get classes
  vector<SprClass> classes;
  data_->classes(classes);

  // exit
  return new SprEmptyFilter(replica,classes,weights,true);
}


SprEmptyFilter* SprBootstrap::weightedReplica(int npts)
{
  // init
  unsigned int size = data_->size();
  if( size == 0 ) return 0;
  if( npts<=0 || npts>static_cast<int>(size) ) npts = size;
  SprData* replica = data_->emptyCopy();

  // init weights
  double wtot = 0;
  vector<double> w;
  data_->weights(w);
  assert( w.size() == size );
  for( unsigned int i=0;i<size;i++ ) wtot += w[i];
  assert( wtot > 0 );
  w[0] /= wtot;
  for( unsigned int i=1;i<size;i++ )
    w[i] = w[i-1] + w[i]/wtot;

  // make array of uniform random numbers on [0,1]
  double* r = new double [npts];
  generator_.sequence(r,npts);
  for( int i=0;i<npts;i++ ) {
    vector<double>::iterator iter = find_if(w.begin(),w.end(),
					    bind2nd(greater<double>(),r[i]));
    unsigned int iuse = iter - w.begin();
    iuse = ( iuse<size ? iuse : size-1 );
    replica->uncheckedInsert((*data_)[iuse]);
  }
  delete [] r;

  // sanity check
  if( static_cast<int>(replica->size()) != npts ) {
    delete replica;
    return 0;
  }

  // get classes
  vector<SprClass> classes;
  data_->classes(classes);

  // exit
  return new SprEmptyFilter(replica,classes,true);
}

