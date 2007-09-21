//$Id: SprAbsTrainedClassifier.cc,v 1.6 2007/07/11 19:52:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <stdio.h>
#include <utility>
#include <fstream>

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


bool SprAbsTrainedClassifier::store(const char* filename) const
{
  // open file for output
  string fname = filename;
  ofstream os(fname.c_str());
  if( !os ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }
 
  // store into file
  this->print(os);

  // store variables
  os << "==================================================" << endl;
  os << "Dimensions:" << endl;
  for( int i=0;i<vars_.size();i++ ) {
    char s [200];
    sprintf(s,"%5i %40s",i,vars_[i].c_str());
    os << s << endl;
  }
  os << "==================================================" << endl;

  // exit
  return true;
}


bool SprAbsTrainedClassifier::storeCode(const char* filename) const
{
  // open file for output
  string fname = filename;
  ofstream os(fname.c_str());
  if( !os ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }

  // store
  return this->generateCode(os);
}
