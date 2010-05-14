#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

int SiStripConfObject::getInt( const std::string & name )
{
  int value = 0;
  get(name, &value);
  return value;
}

double SiStripConfObject::getDouble( const std::string & name )
{
  double value = 0.;
  get(name, &value);
  return value;
}

std::string SiStripConfObject::getString( const std::string & name )
{
  std::string value;
  get(name, &value);  
  return value;
}

void SiStripConfObject::printSummary(std::stringstream & ss) const
{
  parMap::const_iterator it = parameters.begin();
  for( ; it != parameters.end(); ++it ) {
    ss << "parameter name = " << it->first << " value = " << it->second << std::endl;
  }
}

void SiStripConfObject::printDebug(std::stringstream & ss) const
{
  printSummary(ss);
}
