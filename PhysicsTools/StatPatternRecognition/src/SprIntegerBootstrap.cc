//$Id: SprIntegerBootstrap.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"

#include <vector>

using namespace std;


bool SprIntegerBootstrap::replica(std::vector<unsigned>& v, int npts)
{
  // init
  if( npts <= 0 ) npts = nsample_;

  // reset input vector
  v.clear();

  // make array of uniform random numbers on [0,1]
  double* r = new double [npts];
  generator_.sequence(r,npts);
  unsigned iuse = 0;
  for( int i=0;i<npts;i++ ) {
    iuse = unsigned(r[i] * dim_);
    if( iuse < dim_ ) v.push_back(iuse);
  }
  delete [] r;

  // exit
  return (static_cast<int>(v.size())==npts);
}


bool SprIntegerBootstrap::replica(std::set<unsigned>& v, int npts)
{
  // init
  if( npts <= 0 ) npts = nsample_;

  // reset input set
  v.clear();

  // make array of uniform random numbers on [0,1]
  double* r = new double [npts];
  generator_.sequence(r,npts);
  unsigned iuse = 0;
  for( int i=0;i<npts;i++ ) {
    iuse = unsigned(r[i] * dim_);
    if( iuse < dim_ ) v.insert(iuse);
  }
  delete [] r;

  // exit
  return !v.empty();
}



