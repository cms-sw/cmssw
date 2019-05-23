#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <iterator>
#include <regex>

using namespace std;
using namespace cms;
using namespace edm;

string_view
DDSpecPar::strValue(const char* key) const {
  auto const& item = spars.find(key);
  if(item == end(spars))
    return string();
  return *begin(item->second);
}

bool
DDSpecPar::hasValue(const char* key) const {
  if(numpars.find(key) != end(numpars))
    return true;
  else
    return false;
}

double
DDSpecPar::dblValue(const char* key) const {
  auto const& item = numpars.find(key);
  if(item == end(numpars))
    return 0;
  return *begin(item->second);
}

void
DDSpecParRegistry::filter(DDSpecParRefs& refs,
			  string_view attribute, string_view value) const {

  bool found(false);
  for_each(begin(specpars), end(specpars), [&refs, &attribute,
					    &value, &found](const auto& k) {
    found = false;
    for_each(begin(k.second.spars), end(k.second.spars), [&](const auto& l) {
	if(l.first == attribute) {
	  for_each(begin(l.second), end(l.second), [&](const auto& m) {
	      if(m == value)
		found = true;
	    });
	}
      });
    if(found) {
      refs.emplace_back(&k.second);
    }
  });
}

void
DDSpecParRegistry::filter(DDSpecParRefs& refs,
			  string_view attribute) const {

  bool found(false);
  for_each(begin(specpars), end(specpars), [&refs, &attribute,
					    &found](const auto& k) {
    found = false;
    for_each(begin(k.second.spars), end(k.second.spars), [&](const auto& l) {
	if(l.first == attribute) {
	  found = true;
	}
      });
    if(found) {
      refs.emplace_back(&k.second);
    }
  });
}
