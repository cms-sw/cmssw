//$Id: SprClass.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <algorithm>
#include <sstream>

using namespace std;


bool SprClass::operator==(int cls) const 
{
  if( negate_ ) {
    for( unsigned int i=0;i<classes_.size();i++ )
      if( cls == classes_[i] ) return false;
    return true;
  }
  else {
    for( unsigned int i=0;i<classes_.size();i++ )
      if( cls == classes_[i] ) return true;
    return false;
  }
  return false;
}


bool SprClass::operator==(const SprClass& other) const 
{
  if( negate_ != other.negate_ ) return false;
  if( classes_.size() != other.classes_.size() ) return false;
  for( unsigned int i=0;i<classes_.size();i++ ) {
    if( find(other.classes_.begin(),other.classes_.end(),classes_[i]) 
	== other.classes_.end() ) return false;
  }
  for( unsigned int i=0;i<other.classes_.size();i++ ) {
    if( find(classes_.begin(),classes_.end(),other.classes_[i]) 
	== classes_.end() ) return false;
  }
  return true;
}


bool SprClass::checkClasses() const
{
  for( vector<int>::const_iterator iter=classes_.begin();
       iter!=classes_.end();iter++ ) {
    if( find(iter+1,classes_.end(),*iter) != classes_.end() ) {
      cerr << "Class " << *iter << " has been entered twice." << endl;
      return false;
    }
  }
  return true;
}


int SprClass::overlap(const SprClass& other) const
{
  if( negate_ || other.negate_ ) return -1;
  for( unsigned int i=0;i<classes_.size();i++ ) {
    if( find(other.classes_.begin(),other.classes_.end(),classes_[i]) 
	!= other.classes_.end() ) return 1;
  }
  for( unsigned int i=0;i<other.classes_.size();i++ ) {
    if( find(classes_.begin(),classes_.end(),other.classes_[i]) 
	!= classes_.end() ) return 1;
  }
  return 0;
}


std::string SprClass::toString() const
{
  ostringstream os;
  os << *this;
  return os.str();
}
