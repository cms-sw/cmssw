//$Id: SprSimpleReader.cc,v 1.6 2007/07/24 23:05:12 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPreFilter.hh"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;


SprSimpleReader::SprSimpleReader(int mode, SprPreFilter* filter)
  : 
  SprAbsReader(filter), 
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
  vector<string> selected;
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
	ind[varCounter] = selected.size();
	selected.push_back(varName);
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
	  ind[i] = selected.size();
	  selected.push_back(varName);
	}
      }
      break;
    }
  }

  // check if all requested input variables have been found
  for( set<string>::const_iterator i=include_.begin();i!=include_.end();i++ ) {
    if( find(selected.begin(),selected.end(),*i) == selected.end() ) {
      cerr << "Variable " << i->c_str() 
	   << " has not been found in file " << fname.c_str() << endl;
      return 0;
    }
  }

  // set up filter
  if( filter_!=0 && !filter_->setVars(selected) ) {
    cerr << "Unable to apply pre-filter requirements." << endl;
    return 0;
  }

  // get a new list of variables
  vector<string> transformed;
  if( filter_ != 0 ) {
    if( !filter_->transformVars(selected,transformed) ) {
      cerr << "Pre-filter is unable to transform variables." << endl;
      return 0;
    }
  }
  if( transformed.empty() ) transformed = selected; 

  // construct a data object to hold the sample points
  auto_ptr<SprData> data(new SprData);
  if( !data->setVars(transformed) ) {
    cerr << "Unable to set variable list for input data." << endl;
    return 0;
  }

  // read in points, one by one
  vector<double> v(selected.size());
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
    double weight = 1.;
    if( mode_ == 4 || mode_ == 6 ) ist >> weight;

    // copy a new point into SprData
    if( mode_ == 1 || mode_ == 4 || mode_ == 5 || mode_ == 6 || readcls ) {
      int icls = 0;
      ist >> icls;
      readcls = false;
      if( mode_ == 3 ) {
	icls = ( icls<=0 ? -1 : 1 );
	icls = ( (icls*charge)<0 ? 0 : 1);
      }
      if( filter_!=0 && !filter_->pass(icls,v) ) continue;
      if( filter_ != 0 ) {
	vector<double> vNew;
	if( filter_->transformCoords(v,vNew) ) {
	  data->insert(icls,vNew);
	  if( mode_ == 4 || mode_ == 6 ) weights.push_back(weight);
	  continue;
	}
	cerr << "Pre-filter is unable to transform coordinates." << endl;
	return 0;
      }
      data->insert(icls,v);
      if( mode_ == 4 || mode_ == 6 ) weights.push_back(weight);
    }
    else {
      // the next line contains the class for this point
      readcls = true;
    }
  }

  // exit
  if( mode_ == 4 || mode_ == 6 )
    return new SprEmptyFilter(data.release(), weights, true);
  return new SprEmptyFilter(data.release(), true);
}

