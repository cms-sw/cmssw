#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

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
