//$Id: SprBoxFilter.cc,v 1.3 2006/11/13 19:09:41 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBoxFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"

#include <utility>
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;


bool SprBoxFilter::pass(const SprPoint* p) const
{
  bool passed = true;
  for( SprGrid::const_iterator iter=cuts_.begin();
       iter!=cuts_.end();iter++ ) {
    unsigned d = iter->first;
    if( d < p->dim() ) {
      const SprCut& cut = iter->second;
      if( !cut.empty() ) {
	passed = false;
	double r = (p->x_)[d];
	for( int j=0;j<cut.size();j++ ) {
	  if( r>cut[j].first && r<cut[j].second ) {
	    passed = true;
	    break;
	  }
	}
	if( !passed ) return false;
      }
    }
  }
  return true;
}


bool SprBoxFilter::setCut(int i, const SprCut& cut)
{
  // sanity check
  if( i<0 || i>=this->dim() ) {
    cerr << "Index out of dimensionality range " << i 
	 << " " << this->dim() << endl;
    return false;
  }

  // find out if a cut on this dimension exists already
  SprGrid::iterator iter = cuts_.find(i);
  if( iter == cuts_.end() ) {
    cuts_.insert(pair<const unsigned,SprCut>(i,cut));
  }
  else {
    iter->second = cut;
  }

  // exit
  return true;
}


bool SprBoxFilter::setCut(const char* var, const SprCut& cut)
{
  // init
  string svar = var;

  // get a list of variables
  vector<string> vars;
  this->vars(vars);

  // find this one
  vector<string>::const_iterator iter = ::find(vars.begin(),vars.end(),svar);

  // set cut
  if( iter != vars.end() ) {
    int d = iter - vars.begin();
    return this->setCut(d,cut);
  }

  // exit
  cerr << "Variable " << svar << " not found." << endl;
  return false;
}


bool SprBoxFilter::setCut(const std::vector<SprCut>& cuts)
{
  cuts_.clear();
  for( int i=0;i<cuts.size();i++ )
    cuts_.insert(pair<const unsigned,SprCut>(i,cuts[i]));
  return true;
}

