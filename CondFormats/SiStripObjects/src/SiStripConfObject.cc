#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

using namespace std;

bool SiStripConfObject::put( const string & name, const int & value )
{
  names_.push_back(name);
  values_.push_back(value);
  return true;
}

int SiStripConfObject::get( const string & name )
{
  vector<string>::iterator it = find(names_.begin(), names_.end(), name);
  if( it == names_.end() ) {
    edm::LogError("SiStripConfObject::get") << "Error: no parameter associated to " << name << "returning -1" << endl;
    return( -1 );
  }
  return values_[distance( names_.begin(), it )];
}

void SiStripConfObject::printSummary(std::stringstream & ss) const
{
  vector<string>::const_iterator namesIt = names_.begin();
  vector<int>::const_iterator valuesIt = values_.begin();
  for( ; namesIt != names_.end(); ++namesIt, ++valuesIt ) {
    ss << "parameter name = " << *namesIt << " value = " << *valuesIt << endl;
  }
}

void SiStripConfObject::printDebug(std::stringstream & ss) const
{
  printSummary(ss);
}
