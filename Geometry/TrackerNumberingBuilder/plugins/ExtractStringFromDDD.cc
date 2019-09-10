#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <string>

template <>
std::string ExtractStringFromDDD<DDFilteredView>::getString(const std::string &s, DDFilteredView *fv) {
  DDValue val(s);
  std::vector<const DDsvalues_type *> result;
  fv->specificsV(result);
  std::vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it) {
    foundIt = DDfetch(*it, val);
    if (foundIt)
      break;
  }
  if (foundIt) {
    std::vector<std::string> const &temp = val.strings();
    if (temp.size() != 1) {
      throw cms::Exception("Configuration") << " ERROR: I need 1 " << s << " tags";
    }
    return temp[0];
  }
  return "NotFound";
}

template <>
std::string ExtractStringFromDDD<cms::DDFilteredView>::getString(const std::string &s, cms::DDFilteredView *fv) {
  auto result = fv->getString(s);
  return {result.data(), result.size()};
}
