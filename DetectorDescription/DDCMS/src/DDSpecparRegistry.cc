#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

using namespace std;
using namespace tbb;
using namespace cms;

void
DDSpecParRegistry::filter(DDSpecParRefMap& refToMap,
			  string_view attribute, string_view value) const {

  bool found(false);
  for_each(begin(specpars), end(specpars), [&refToMap, &attribute,
					    &value, &found, this](const auto& k) {
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
      refToMap.emplace(&k.first, &k.second);
    }
  });
}
