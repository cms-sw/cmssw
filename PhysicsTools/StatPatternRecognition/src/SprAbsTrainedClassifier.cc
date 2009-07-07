//$Id: SprAbsTrainedClassifier.cc,v 1.3 2007/11/12 06:19:18 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <stdio.h>
#include <utility>
#include <fstream>

using namespace std;


bool SprAbsTrainedClassifier::accept(const std::vector<double>& v, 
				     double& response) const
{
  response = this->response(v);
  if( cut_.empty() ) return true;
  bool passed = false;
  for( unsigned int i=0;i<cut_.size();i++ ) {
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
  for( unsigned int i=0;i<vars_.size();i++ ) {
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
