//$Id: SprAbsMultiClassLearner.cc,v 1.1 2007/05/14 18:08:08 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"

#include <fstream>
#include <string>
#include <vector>

using namespace std;


bool SprAbsMultiClassLearner::store(const char* filename) const
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
  vector<string> vars;
  data_->vars(vars);
  assert( vars.size() == data_->dim() );
  os << "==================================================" << endl;
  os << "Dimensions:" << endl;
  for( int i=0;i<vars.size();i++ ) {
    char s [200];
    sprintf(s,"%5i %40s",i,vars[i].c_str());
    os << s << endl;
  }
  os << "==================================================" << endl;

  // exit
  return true;
}
