//$Id: SprTrainedFisher.cc,v 1.5 2007/02/05 21:49:46 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"

#include <iomanip>

using namespace std;


double SprTrainedFisher::response(const std::vector<double>& v) const
{
  // sanity check
  int size = v.size();
  if( size != linear_.num_row() ) {
    cerr << "Input vector and Fisher vector have unmatching dimensions! " 
	 << size << " " << linear_.num_row() << endl;
    return 0;
  }

  // compute linear contribution
  double d = 0;
  for( int i=0;i<size;i++ ) {
    d += v[i] * linear_[i];
  }

  // compute quadratic contribution
  if( this->mode() == 2 ) {
    if( size != quadr_.num_row() ) {
      cerr << "Input vector and Fisher matrix have unmatching dimensions! " 
	   << size << " " << quadr_.num_row() << endl;
      return 0;
    }
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
  if( v.num_row() != linear_.num_row() ) {
    cerr << "Input vector and Fisher vector have unmatching dimensions! " 
	 << v.num_row() << " " << linear_.num_row() << endl;
    return 0;
  }

  // compute linear contribution
  double d = dot(v,linear_);

  // compute quadratic contribution
  if( this->mode() == 2 ) {
    if( v.num_row() != quadr_.num_row() ) {
      cerr << "Input vector and Fisher matrix have unmatching dimensions! " 
	   << v.num_row() << " " << quadr_.num_row() << endl;
      return 0;
    }
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
  os << "Trained Fisher" << endl;
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
