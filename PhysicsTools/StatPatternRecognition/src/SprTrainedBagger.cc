//$Id: SprTrainedBagger.cc,v 1.3 2007/10/30 18:56:14 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <stdio.h>
#include <cassert>

using namespace std;


SprTrainedBagger::SprTrainedBagger(const std::vector<
		   std::pair<const SprAbsTrainedClassifier*,bool> >& 
		   trained, bool discrete) 
  : 
  SprAbsTrainedClassifier(),
  trained_(trained),
  discrete_(discrete)
{
  assert( !trained_.empty() );
  this->setCut(SprUtils::lowerBound(0.5));
}


SprTrainedBagger::SprTrainedBagger(const SprTrainedBagger& other)
  :
  SprAbsTrainedClassifier(other),
  trained_(),
  discrete_(other.discrete_)
{
  for( unsigned int i=0;i<other.trained_.size();i++ )
    trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>
		       (other.trained_[i].first->clone(),true));
}


double SprTrainedBagger::response(const std::vector<double>& v) const
{
  // init
  double r = 0;

  // discrete/continuous
  if( discrete_ ) {
    int out = 0;
    for( unsigned int i=0;i<trained_.size();i++ )
      out += ( trained_[i].first->accept(v) ? 1 : -1 );
    r = out;
    r /= 2.*trained_.size();
    r += 0.5;
  }
  else {
    for( unsigned int i=0;i<trained_.size();i++ )
      r += trained_[i].first->response(v);
    r /= trained_.size();
  }

  // exit
  return r;
}


void SprTrainedBagger::destroy()
{
  for( unsigned int i=0;i<trained_.size();i++ ) {
    if( trained_[i].second )
      delete trained_[i].first;
  }
}


void SprTrainedBagger::print(std::ostream& os) const
{
  os << "Trained Bagger " << SprVersion << endl;
  os << "Classifiers: " << trained_.size() << endl;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    os << "Classifier " << i 
       << " " << trained_[i].first->name().c_str() << endl;
    trained_[i].first->print(os);
  }
}


bool SprTrainedBagger::generateCode(std::ostream& os) const 
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


SprTrainedBagger& SprTrainedBagger::operator+=(const SprTrainedBagger& other)
{
  // check vars
  if( vars_.size() != other.vars_.size() ) {
    cerr << "Unable to add Bagger: variable lists do not match." << endl;
    return *this;
  }
  for( unsigned int i=0;i<vars_.size();i++ ) {
    if( vars_[i] != other.vars_[i] ) {
      cerr << "Unable to add Bagger: variable lists do not match." << endl;
      cerr << "Variables " << i << ": " 
	   << vars_[i] << " " << other.vars_[i] << endl;
      return *this;
    }
  }

  // check discreteness
  if( discrete_ != other.discrete_ ) {
    cerr << "Unable to add Bagger: discreteness does not match." << endl;
    return *this;
  }

  // add
  for( unsigned int i=0;i<other.trained_.size();i++ ) {
    trained_.push_back(pair<const SprAbsTrainedClassifier*,
		       bool>(other.trained_[i].first->clone(),true));
  }
  this->setCut(SprUtils::lowerBound(0.5));

  // exit
  return *this;
}


const SprTrainedBagger operator+(const SprTrainedBagger& l,
				 const SprTrainedBagger& r)
{
  // check variable list
  assert( l.vars_.size() == r.vars_.size() );
  for( unsigned int i=0;i<l.vars_.size();i++ )
    assert( l.vars_[i] == r.vars_[i] );

  // add classifiers
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained;
  for( unsigned int i=0;i<l.trained_.size();i++ ) {
    trained.push_back(pair<const SprAbsTrainedClassifier*,
		      bool>(l.trained_[i].first->clone(),true));
  }
  
  for( unsigned int i=0;i<r.trained_.size();i++ ) {
    trained.push_back(pair<const SprAbsTrainedClassifier*,
		      bool>(r.trained_[i].first->clone(),true));
  }

  // add discrete
  assert( l.discrete_ == r.discrete_ );

  // make bagger and set cut
  SprTrainedBagger newBagger(trained,l.discrete_);
  newBagger.setCut(SprUtils::lowerBound(0.5));

  // exit
  return newBagger;
}
