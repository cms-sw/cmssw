//$Id: SprSimpleReader.cc,v 1.3 2006/11/13 19:09:43 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;


SprSimpleReader::SprSimpleReader(int mode)
  : 
  SprAbsReader(), 
  mode_(mode),
  include_(),
  exclude_()
{
  assert( mode_>0 && mode_<7 );
}


SprAbsFilter* SprSimpleReader::read(const char* filename)
{
  // cannot request and exclude variables at the same time
  if( !include_.empty() && !exclude_.empty() ) {
    cerr << "You cannot include and exclude variables at the same time." 
	 << endl;
    return 0;
  }

  // open file
  string fname = filename;
  ifstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return 0;
  }

  // read number of dimensions
  string line;
  unsigned dim = 0;
  unsigned nline = 0;
  while( getline(file,line) ) {
    nline++;
    if( line.find('#') != string::npos )
      line.erase( line.find_first_of('#') );
    if( line.find_first_not_of(' ') == string::npos ) continue;
    istringstream ist(line);
    ist >> dim;
    assert( dim != 0 );
    break;
  }

  // read var names
  vector<int> ind(dim,-1);
  vector<string> sorted;
  int varCounter = 0;
  while( getline(file,line) ) {
    nline++;
    if( line.find('#') != string::npos )
      line.erase( line.find_first_of('#') );
    if( line.find_first_not_of(' ') == string::npos ) continue;
    if( mode_==5 || mode_==6 ) {// variable names take separate lines
      istringstream ist(line);
      std::string varName;
      ist >> varName;
      if( exclude_.find(varName)==exclude_.end() &&
	  (include_.empty() || include_.find(varName)!=include_.end()) ) {
	ind[varCounter] = sorted.size();
	sorted.push_back(varName);
      }
      if( ++varCounter >= dim ) break;
    }
    else {// all variable names are on one line
      istringstream ist(line);
      for( int i=0;i<dim;i++ ) {
	std::string varName;
	ist >> varName;
	if( exclude_.find(varName)==exclude_.end() &&
	    (include_.empty() || include_.find(varName)!=include_.end()) ) {
	  ind[i] = sorted.size();
	  sorted.push_back(varName);
	}
      }
      break;
    }
  }

  // check if all requested input variables have been found
  for( set<string>::const_iterator i=include_.begin();i!=include_.end();i++ ) {
    if( find(sorted.begin(),sorted.end(),*i) == sorted.end() ) {
      cerr << "Variable " << i->c_str() 
	   << " has not been found in file " << fname.c_str() << endl;
      return 0;
    }
  }

  // construct a data object to hold the sample points
  auto_ptr<SprData> data(new SprData);
  data->setVars(sorted);
 
  // read in points, one by one
  vector<double> v(sorted.size());
  vector<double> weights;
  int charge = 0;
  bool readcls = false;
  while( getline(file,line) ) {
    nline++;
    if( line.find('#') != string::npos )
      line.erase( line.find_first_of('#') );
    if( line.find_first_not_of(' ') == string::npos ) continue;
    istringstream ist(line);
    if( !readcls ) {
      for( int i=0;i<dim;i++ ) {
	double r = 0;
	ist >> r;
	int index = ind[i];
	if( index >= 0 ) v[index] = r;
      }
      if( mode_ == 3 ) ist >> charge;
    }
    if( mode_ == 4 || mode_ == 6 ) {
      double weight = 0;
      ist >> weight;
      weights.push_back(weight);
    }
    if( mode_ == 1 || mode_ == 4 || mode_ == 5 || mode_ == 6 || readcls ) {
      int icls = 0;
      ist >> icls;
      readcls = false;
      if( mode_ == 3 ) {
	icls = ( icls<=0 ? -1 : 1 );
	icls = ( (icls*charge)<0 ? 0 : 1);
      }
      data->insert(icls,v);
    }
    else {
      // note that the next line contains the class for this point
      readcls = true;
    }
  }

  // exit
  if( mode_ == 4 || mode_ == 6 )
    return new SprEmptyFilter(data.release(), weights, true);
  return new SprEmptyFilter(data.release(), true);
}

