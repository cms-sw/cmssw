// $Id: SprAbsVarTransformer.cc,v 1.1 2007/11/12 06:19:18 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;


bool SprAbsVarTransformer::store(const char* filename) const
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

  // store old variables
  os << "==================================================" << endl;
  os << "Old Variables:" << endl;
  for( unsigned int i=0;i<oldVars_.size();i++ ) {
    char s [200];
    sprintf(s,"%5i %40s",i,oldVars_[i].c_str());
    os << s << endl;
  }
  os << "==================================================" << endl;

  // store new variables
  os << "==================================================" << endl;
  os << "New Variables:" << endl;
  for( unsigned int i=0;i<newVars_.size();i++ ) {
    char s [200];
    sprintf(s,"%5i %40s",i,newVars_[i].c_str());
    os << s << endl;
  }
  os << "==================================================" << endl;

  // exit
  return true;
}
