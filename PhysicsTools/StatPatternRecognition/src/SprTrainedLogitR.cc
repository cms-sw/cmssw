//$Id: SprTrainedLogitR.cc,v 1.4 2007/02/05 21:49:46 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"

#include <iomanip>

using namespace std;


double SprTrainedLogitR::response(const std::vector<double>& v) const
{
  // sanity check
  int size = v.size();
  if( size != beta_.num_row() ) {
    cerr << "Input vector and LogitR have unmatching dimensions! " 
	 << size << " " << beta_.num_row() << endl;
    return 0;
  }

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
  if( v.num_row() != beta_.num_row() ) {
    cerr << "Input vector and LogitR have unmatching dimensions! " 
	 << v.num_row() << " " << beta_.num_row() << endl;
    return 0;
  }

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
  os << "Trained LogitR" << endl;
  os << "LogitR dimensionality: " << beta_.num_row() << endl;
  os << "LogitR response: L = Beta0 + Beta*X" << endl;  
  os << "By default logit transform is applied: L <- 1/[1+exp(-L)]" << endl;
  os << "Beta0: " << beta0_ << endl;
  os << "Vector of Beta Coefficients:" << endl;
  for( int i=0;i<beta_.num_row();i++ )
    os << setw(10) << beta_[i] << " ";
  os << endl;
}
