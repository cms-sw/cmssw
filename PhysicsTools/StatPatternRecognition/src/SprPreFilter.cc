//$Id: SprPreFilter.cc,v 1.2 2007/10/30 18:56:14 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPreFilter.hh"

#include <cassert>
#include <algorithm>
#include <iterator>
#include <iostream>

using namespace std;


bool SprPreFilter::resetSelection()
{
  userSelectionVars_.clear();
  userSelection_ = 0;
  selectionVarToIndex_.clear();
  userClasses_.clear();
  return true;
}


bool SprPreFilter::resetTransform()
{
  userTransformInputVars_.clear();
  userTransformOutputVars_.clear();
  userTransform_ = 0;
  transformVarToIndex_.clear();
  return true;
}


bool SprPreFilter::resetClass()
{
  userClassVars_.clear();
  userClassDefinition_ = 0;
  classVarToIndex_.clear();
  return true;
}


bool SprPreFilter::setVars(const std::vector<std::string>& vars) 
{
  // check selection vars
  if( !this->setVarIndex(vars,userSelectionVars_,selectionVarToIndex_) ) {
    cerr << "Unable to set selection variables in SprPreFilter." << endl;
    this->resetSelection();
    return false;
  }

  // check transform vars
  if( !this->setVarIndex(vars,userTransformInputVars_,transformVarToIndex_) ) {
    cerr << "Unable to set transformation variables in SprPreFilter." << endl;
    this->resetTransform();
    return false;
  }

  // check class vars
  if( !this->setVarIndex(vars,userClassVars_,classVarToIndex_) ) {
    cerr << "Unable to set class variables in SprPreFilter." << endl;
    this->resetClass();
    return false;
  }

  // exit
  return true;
}


bool SprPreFilter::setVarIndex(const std::vector<std::string>& dataVars,
			       const std::vector<std::string>& userVars,
			       std::vector<int>& indexMap)
{
  // set vars
  if( dataVars.empty() ) {
    cerr << "No data variables specified for SprPreFilter." << endl;
    return false;
  }

   // get a list of requested variables
  if( userVars.empty() ) {
    cout << "No user variables specified for SprPreFilter." << endl;
    return true;
  }

  // match variables
  int nVars = userVars.size();
  indexMap.resize(nVars);
  for( int i=0;i<nVars;i++ ) {
    vector<string>::const_iterator iter 
      = find(dataVars.begin(),dataVars.end(),userVars[i]);
    if( iter == dataVars.end() ) {
      cerr << "Unknown variable requested by the user pre-filter: "
           << userVars[i] << endl;
      return false;
    }
    indexMap[i] = iter - dataVars.begin();
  }

  // exit
  return true;
}


bool SprPreFilter::setSelection(SprPreVars userVars,
				SprPreSelection userSelection,
				SprPreClasses userClasses)
{
  // supplying null pointers is equivalent to removing all reqs
  if( userVars==0 || userSelection==0 ) {
    cout << "No selection requirements will be applied by SprPreFilter."
         << endl;
    return this->resetSelection();
  }

  // set selection and vars
  userSelectionVars_ = userVars();
  userSelection_ = userSelection;
  userClasses_ = userClasses();

  // exit
  return true;
}


bool SprPreFilter::setTransform(SprPreVars inputVars,
				SprPreVars outputVars,
				SprPreTransform userTransform)
{
  // supplying null pointers is equivalent to removing all reqs
  if( inputVars==0 || outputVars==0 || userTransform==0 ) {
    cout << "No transformation will be applied by SprPreFilter."
         << endl;
    return this->resetTransform();
  }

  // set selection and vars
  userTransformInputVars_  = inputVars();
  userTransformOutputVars_ = outputVars();
  userTransform_ = userTransform;

  // exit
  return true;
}


bool SprPreFilter::setClass(SprPreVars userVars,
			    SprPreClassDefinition classDefinition)
{
  // supplying null pointers is equivalent to removing all reqs
  if( userVars==0 || classDefinition==0 ) {
    cout << "No class requirements will be applied by SprPreFilter."
         << endl;
    return this->resetClass();
  }

  // set selection and vars
  userClassVars_ = userVars();
  userClassDefinition_ = classDefinition;

  // exit
  return true;
}


bool SprPreFilter::pass(int icls, const std::vector<double>& input) const
{
  // if no user selection defined, accept
  if( userClasses_.empty() && 
      (selectionVarToIndex_.empty() || userSelection_==0) ) return true;

  // check class
  if( find(userClasses_.begin(),userClasses_.end(),icls)
      == userClasses_.end() ) return false;

  // get vector of matched coordinates
  unsigned int nVars = selectionVarToIndex_.size();
  vector<double> v(nVars);
  for( unsigned int i=0;i<nVars;i++ ) {
    int unsigned index = selectionVarToIndex_[i];
    assert( index < input.size() );
    v[i] = input[index];
  }

  // run the user filter
  return userSelection_(v);
}


bool SprPreFilter::transformCoords(const std::vector<double>& input, 
				   std::vector<double>& output) const
{
  // if no user transform defined, accept
  if( transformVarToIndex_.empty() || userTransform_==0 ) return true;

  // get vector of matched coordinates
  int nVars = transformVarToIndex_.size();
  vector<double> v(nVars);
  for( int i=0;i<nVars;i++ ) {
    unsigned int index = transformVarToIndex_[i];
    assert( index < input.size() );
    v[i] = input[index];
  }

  // transform
  vector<double> vNew;
  userTransform_(v,vNew);
  if( vNew.size() != userTransformOutputVars_.size() ) {
    cerr << "Dimensionality of transformed coordinates and " 
	 << "variables do not match: " 
	 << vNew.size() << " " << userTransformOutputVars_.size() << endl;
    return false;
  }

  // copy the input vector over skipping the transformed values
  int jstart = 0;
  output.clear();
  for( int i=0;i<nVars;i++ ) {
    for( int j=jstart;j< static_cast<int>(input.size()) ;j++ ) {
      if( j == transformVarToIndex_[i] ) {
	jstart = j+1;
	break;
      }
      output.push_back(input[j]);
    }
  }
  assert( output.size() == (input.size()-transformVarToIndex_.size()) );

  // append the new vector to the copied one
  copy(vNew.begin(),vNew.end(),back_inserter(output));

  // exit
  return true;
}


bool SprPreFilter::transformVars(const std::vector<std::string>& input, 
				 std::vector<std::string>& output) const
{
  // if no user transform defined, accept
  if( transformVarToIndex_.empty() || userTransformOutputVars_.empty() ) 
    return true;

  // copy the input vector over skipping the transformed values
  int nVars = transformVarToIndex_.size();
  int jstart = 0;
  output.clear();
  for( int i=0;i<nVars;i++ ) {
    for( int j=jstart; j < static_cast<int>(input.size());j++ ) {
      if( j == transformVarToIndex_[i] ) {
	jstart = j+1;
	break;
      }
      output.push_back(input[j]);
    }
  }
  assert( output.size() == (input.size()-transformVarToIndex_.size()) );

  // append the new vector to the copied one
  copy(userTransformOutputVars_.begin(),userTransformOutputVars_.end(),
       back_inserter(output));

  // exit
  return true;
}


std::pair<int,bool> SprPreFilter::computeClass(
				      const std::vector<double>& input) const
{
  // if no user class is defined, return 0
  if( classVarToIndex_.empty() || userClassDefinition_==0 ) 
    return pair<int,bool>(0,false);

  // get vector of matched coordinates
  int nVars = classVarToIndex_.size();
  vector<double> v(nVars);
  for( int i=0;i<nVars;i++ ) {
    unsigned int index = classVarToIndex_[i];
    assert( index < input.size() );
    v[i] = input[index];
  }

  // run the user filter
  return pair<int,bool>(userClassDefinition_(v),true);
}
