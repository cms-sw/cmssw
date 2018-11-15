#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <string>

using namespace cms;

std::string ExtractStringFromDDD::getString(std::string const & s,DDFilteredView* fv){ 
  DDValue val(s);
  std::vector<const DDsvalues_type *> result;
  fv->specificsV(result);
  std::vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it)   {    
    foundIt = DDfetch(*it,val);
    if (foundIt) break;
  }    
  if (foundIt)   { 
    std::vector<std::string> const & temp = val.strings();
    if (temp.size() != 1) {
     throw cms::Exception("Configuration")<< " ERROR: I need 1 "<< s << " tags";
    }
    return temp[0]; 
  }
  return "NotFound";
}
