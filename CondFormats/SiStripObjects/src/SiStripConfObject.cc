#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

bool SiStripConfObject::put( const std::string & name, const int & value )
{
  names.push_back(name);
  values.push_back(value);
  return true;
}

int SiStripConfObject::get( const std::string & name )
{
  std::vector<std::string>::iterator it = std::find(names.begin(), names.end(), name);
  if( it == names.end() ) {
    edm::LogError("SiStripConfObject::get") << "Error: no parameter associated to " << name << " returning -1" << std::endl;
    return( -1 );
  }
  return values[distance( names.begin(), it )];
}

void SiStripConfObject::printSummary(std::stringstream & ss) const
{
  std::vector<std::string>::const_iterator namesIt = names.begin();
  std::vector<int>::const_iterator valuesIt = values.begin();
  for( ; namesIt != names.end(); ++namesIt, ++valuesIt ) {
    ss << "parameter name = " << *namesIt << " value = " << *valuesIt << std::endl;
  }
}

void SiStripConfObject::printDebug(std::stringstream & ss) const
{
  printSummary(ss);
}
