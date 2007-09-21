//$Id: SprTrainedFisher.cc,v 1.7 2007/05/15 00:08:26 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <iomanip>
#include <cassert>

using namespace std;


double SprTrainedFisher::response(const std::vector<double>& v) const
{
  // sanity check
  int size = v.size();
  assert( size == linear_.num_row() );

  // compute linear contribution
  double d = 0;
  for( int i=0;i<size;i++ ) {
    d += v[i] * linear_[i];
  }

  // compute quadratic contribution
  if( this->mode() == 2 ) {
    assert( size == quadr_.num_row() );
    double row = 0;
    for( int i=1;i<size;i++ ) {
      row = 0;
      for( int j=0;j<i;j++ )
	row += quadr_[i][j] * v[j];
      d += v[i] * row;
    }
    d *= 2;
    for( int i=0;i<size;i++ )
      d += v[i]*v[i] * quadr_[i][i];
  }

  // add const term
  d += cterm_;

  // apply transform
  if( !standard_ )
    d = SprTransformation::logit(d);

  // exit
  return d;
}


double SprTrainedFisher::response(const SprVector& v) const
{
  // sanity check
  assert( v.num_row() == linear_.num_row() );

  // compute linear contribution
  double d = dot(v,linear_);

  // compute quadratic contribution
  if( this->mode() == 2 ) {
    assert( v.num_row() == quadr_.num_row() );
    d += dot(v,quadr_*v);
  }

  // add const term
  d += cterm_;

  // apply transform
  if( !standard_ )
    d = SprTransformation::logit(d);

  // exit
  return d;
}


void SprTrainedFisher::print(std::ostream& os) const
{
  os << "Trained Fisher " << SprVersion << endl;
  os << "Fisher dimensionality: " << linear_.num_row() << endl;
  os << "Fisher response: F = C + T(L)*X + T(X)*Q*X; T is transposition" 
     << endl;
  os << "By default logit transform is applied: F <- 1/[1+exp(-F)]" << endl;
  os << "Fisher order: " << (quadr_.num_row()>0 ? 2 : 1) << endl;
  os << "Const term: " << cterm_ << endl;
  os << "Linear Part:" << endl;
  for( int i=0;i<linear_.num_row();i++ )
    os << setw(10) << linear_[i] << " ";
  os << endl;
  if( quadr_.num_row() > 0 ) {
    os << "Quadratic Part:" << endl;
    for( int i=0;i<quadr_.num_row();i++ ) {
      for( int j=0;j<quadr_.num_col();j++ ) {
        os << setw(10) << quadr_[i][j] << " ";
      }
      os << endl;
    }
  }
}
