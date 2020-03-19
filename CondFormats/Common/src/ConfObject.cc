#include "CondFormats/Common/interface/ConfObject.h"

void ConfObject::printSummary(std::stringstream& ss) const {
  parMap::const_iterator it = parameters.begin();
  for (; it != parameters.end(); ++it) {
    ss << "parameter name = " << it->first << " value = " << it->second << std::endl;
  }
}

void ConfObject::printDebug(std::stringstream& ss) const { printSummary(ss); }
