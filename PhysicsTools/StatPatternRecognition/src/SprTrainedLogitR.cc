//$Id: SprTrainedLogitR.cc,v 1.6 2007/05/15 00:08:26 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <iomanip>
#include <cassert>

using namespace std;


double SprTrainedLogitR::response(const std::vector<double>& v) const
{
  // sanity check
  int size = v.size();
  assert( size == beta_.num_row() );

  // compute linear contribution
  double result = 0;
  for( int i=0;i<size;i++ ) {
    result += v[i] * beta_[i];
  }

  // add const term
  result += beta0_;

  // transform
  if( !standard_ ) 
    result = SprTransformation::logit(result);

  // exit
  return result;
}


double SprTrainedLogitR::response(const SprVector& v) const
{
  // sanity check
  assert( v.num_row() == beta_.num_row() );

  // compute linear contribution
  double result = dot(v,beta_);

  // add const term
  result += beta0_;

  // transform
  if( !standard_ ) 
    result = SprTransformation::logit(result);

  // exit
  return result;
}


void SprTrainedLogitR::print(std::ostream& os) const
{
  os << "Trained LogitR " << SprVersion << endl;
  os << "LogitR dimensionality: " << beta_.num_row() << endl;
  os << "LogitR response: L = Beta0 + Beta*X" << endl;  
  os << "By default logit transform is applied: L <- 1/[1+exp(-L)]" << endl;
  os << "Beta0: " << beta0_ << endl;
  os << "Vector of Beta Coefficients:" << endl;
  for( int i=0;i<beta_.num_row();i++ )
    os << setw(10) << beta_[i] << " ";
  os << endl;
}
