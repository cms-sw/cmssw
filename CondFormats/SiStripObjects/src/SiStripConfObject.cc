#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

using namespace std;

bool SiStripConfObject::put( const string & name, const int & value )
{
  names.push_back(name);
  values.push_back(value);
  return true;
}

int SiStripConfObject::get( const string & name )
{
  vector<string>::iterator it = find(names.begin(), names.end(), name);
  if( it == names.end() ) {
    edm::LogError("SiStripConfObject::get") << "Error: no parameter associated to " << name << " returning -1" << endl;
    return( -1 );
  }
  return values[distance( names.begin(), it )];
}

void SiStripConfObject::printSummary(std::stringstream & ss) const
{
  vector<string>::const_iterator namesIt = names.begin();
  vector<int>::const_iterator valuesIt = values.begin();
  for( ; namesIt != names.end(); ++namesIt, ++valuesIt ) {
    ss << "parameter name = " << *namesIt << " value = " << *valuesIt << endl;
  }
}

void SiStripConfObject::printDebug(std::stringstream & ss) const
{
  printSummary(ss);
}
