//$Id: SprTrainedAdaBoost.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <stdio.h>
#include <algorithm>
#include <cmath>

using namespace std;


SprTrainedAdaBoost::SprTrainedAdaBoost(const std::vector<
		       std::pair<const SprAbsTrainedClassifier*,bool> >& 
				       trained, 
				       const std::vector<double>& beta,
				       bool useStandard,
				       AdaBoostMode mode) 
  : 
  SprAbsTrainedClassifier(),
  trained_(trained),
  beta_(beta),
  mode_(mode),
  standard_(useStandard),
  epsilon_(0.01)
{
  assert( trained_.size() == beta_.size() );
  assert( !trained_.empty() );
}


SprTrainedAdaBoost::SprTrainedAdaBoost(const SprTrainedAdaBoost& other)
  :
  SprAbsTrainedClassifier(other),
  trained_(),
  beta_(other.beta_),
  mode_(other.mode_),
  standard_(other.standard_),
  epsilon_(other.epsilon_)
{
  for( unsigned int i=0;i<other.trained_.size();i++ )
    trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>
		       (other.trained_[i].first->clone(),true));
}


double SprTrainedAdaBoost::response(const std::vector<double>& v) const
{
  // compute output
  double result = 0;
  if(      mode_==Discrete || mode_==Epsilon ) {
    for( unsigned int i=0;i<beta_.size();i++ ) {
      int out = ( trained_[i].first->accept(v) ? 1 : -1 );
      result += out*beta_[i];
    }
  }
  else if( mode_==Real ) {
    double resp = 0;
    for( unsigned int i=0;i<beta_.size();i++ ) {
      resp = trained_[i].first->response(v);
      resp += (1.-2.*resp)*epsilon_;
      if( resp < SprUtils::eps() ) {
	if( standard_ )
	  return -SprUtils::max();
	else
	  return 0;
      }
      if( resp > 1.-SprUtils::eps() ) {
	if( standard_ )
	  return SprUtils::max();
	else
	  return 1;
      }
      result += SprTransformation::logitHalfInverse(resp)*beta_[i];
    }
  }

  // transform to [0,1] if required
  if( !standard_ )
    result = SprTransformation::logitDouble(result);

  // exit
  return result;
}


void SprTrainedAdaBoost::destroy()
{
  for( unsigned int i=0;i<trained_.size();i++ ) {
    if( trained_[i].second )
      delete trained_[i].first;
  }
}


void SprTrainedAdaBoost::print(std::ostream& os) const
{
  assert( beta_.size() == trained_.size() );
  os << "Trained AdaBoost " << SprVersion << endl;
  os << "Classifiers: " << trained_.size() << endl;
  os << "Mode: " << int(mode_) << "   Epsilon: " << epsilon_ << endl;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    char s [200];
    sprintf(s,"Classifier %6i %s Beta: %12.10f",
	    i,trained_[i].first->name().c_str(),beta_[i]);
    os << s << endl;
  }
  os << "Classifiers:" << endl;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    os << "Classifier " << i 
       << " " << trained_[i].first->name().c_str() << endl;
    trained_[i].first->print(os);
  }
}


bool SprTrainedAdaBoost::generateCode(std::ostream& os) const 
{ 
  // generate weak classifiers
  for( unsigned int i=0;i<trained_.size();i++ ) { 
    string name = trained_[i].first->name();
    os << " // Classifier " << i  
       << " \"" << name.c_str() << "\"" << endl; 
    if( !trained_[i].first->generateCode(os) ) {
      cerr << "Unable to generate code for classifier " << name.c_str() 
	   << endl;
      return false;
    }
    if( i < trained_.size()-1 ) os << endl; 
  }

  // exit
  return true;
} 
