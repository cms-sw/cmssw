//$Id: SprSimpleReader.cc,v 1.5 2008/11/26 22:59:20 elmer Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPreFilter.hh"

#include <algorithm>
#include <utility>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>

using namespace std;


SprSimpleReader::SprSimpleReader(int mode, SprPreFilter* filter)
  : 
  SprAbsReader(filter), 
  mode_(mode),
  include_(),
  exclude_()
{
  assert( mode_>0 && mode_<8 );
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

  // init
  string line;
  unsigned dim = 0;
  unsigned nline = 0;

  // read number of dimensions
  if( mode_ != 7 ) {
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
  }

  // read var names
  vector<int> ind;
  vector<string> selected;
  while( getline(file,line) ) {
    nline++;
    if( line.find('#') != string::npos )
      line.erase( line.find_first_of('#') );
    if( line.find_first_not_of(' ') == string::npos ) continue;
    istringstream ist(line);
    string varName;
    if(      mode_==5 || mode_==6 ) {// variable names take separate lines
      ist >> varName;
      if( exclude_.find(varName)==exclude_.end() &&
	  (include_.empty() || include_.find(varName)!=include_.end()) ) {
	ind.push_back(selected.size());
	selected.push_back(varName);
      }
      if( ind.size() >= dim ) break;
    }
    else if( mode_ == 7 ) {// don't know how many vars yet
      int varCounter = 0;
      while( ist >> varName ) {
	if( ++varCounter > 3 ) {
	  if( exclude_.find(varName)==exclude_.end() &&
	      (include_.empty() || include_.find(varName)!=include_.end()) ) {
	    ind.push_back(selected.size());
	    selected.push_back(varName);
	  }
	}
      }
      dim = selected.size();
      break;
    }
    else {// all variable names are on one line
      ind.clear();
      ind.resize(dim,-1);
      for( unsigned int i=0;i<dim;i++ ) {
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
    // get line
    nline++;
    if( line.find('#') != string::npos )
      line.erase( line.find_first_of('#') );
    if( line.find_first_not_of(' ') == string::npos ) continue;

    // read coords
    int icls = 0;
    double weight = 1.;
    int pointIndex = -1;
    istringstream ist(line);
    if( mode_ == 7 ) {
      ist >> pointIndex >> icls >> weight;
      assert( pointIndex >= 0 );
    }
    if( !readcls ) {
      for( unsigned int i=0;i<dim;i++ ) {
	double r = 0;
	ist >> r;
	int index = ind[i];
	if( index >= 0 ) v[index] = r;
      }
      if( mode_ == 3 ) ist >> charge;
    }
    if( mode_ == 4 || mode_ == 6 ) ist >> weight;

    // if 2 modes split into lines, skip the rest
    if( (mode_==2 || mode_==3) && !readcls ) {
      readcls = true;
      continue;
    }

    // read class
    if( mode_ != 7 ) {
      ist >> icls;
      readcls = false;
    }

    // assign class for special modes
    if( mode_ == 3 ) {
      icls = ( icls<=0 ? -1 : 1 );
      icls = ( (icls*charge)<0 ? 0 : 1);
    }

    // passes selection requirements?
    if( filter_!=0 && !filter_->pass(icls,v) ) continue;

    // compute user-defined class
    if( filter_!=0 ) {
      pair<int,bool> computedClass = filter_->computeClass(v);
      if( computedClass.second ) 
	icls = computedClass.first;
    }

    // transform coordinates
    if( filter_ != 0 ) {
      vector<double> vNew;
      if( filter_->transformCoords(v,vNew) ) {
	if( mode_ == 7 )
	  data->insert(pointIndex,icls,vNew);
	else
	  data->insert(icls,vNew);
      }
      else {
	cerr << "Pre-filter is unable to transform coordinates." << endl;
	return 0;
      }
    }
    else {
      if( mode_ == 7 )
	data->insert(pointIndex,icls,v);
      else
	data->insert(icls,v);
    }

    // store weight
    if( mode_==4 || mode_==6 || mode_==7 ) weights.push_back(weight);
  }

  // exit
  if( mode_ == 4 || mode_ == 6 || mode_==7 )
    return new SprEmptyFilter(data.release(), weights, true);
  return new SprEmptyFilter(data.release(), true);
}

