//$Id: SprAbsTrainedClassifier.cc,v 1.4 2007/02/05 21:49:45 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <stdio.h>
#include <utility>

using namespace std;


/*
  This is, of course, inefficient. I will deal with this later. -IN, Febr 2005
*/
double SprAbsTrainedClassifier::response(const SprVector& v) const
{
  vector<double> vv;
  for( int i=0;i<v.num_row();i++ )
    vv.push_back(v[i]);
  return this->response(vv);
}


bool SprAbsTrainedClassifier::accept(const SprVector& v, double& response) 
  const
{
  vector<double> vv;
  for( int i=0;i<v.num_row();i++ )
    vv.push_back(v[i]);
  return this->accept(vv,response);
}


bool SprAbsTrainedClassifier::accept(const std::vector<double>& v, 
				     double& response) const
{
  response = this->response(v);
  if( cut_.empty() ) return true;
  bool passed = false;
  for( int i=0;i<cut_.size();i++ ) {
    const pair<double,double>& lims = cut_[i];
    if( response>lims.first && response<lims.second ) {
      passed = true;
      break;
    }
  }
  return passed;
}
