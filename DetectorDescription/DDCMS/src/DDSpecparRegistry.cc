#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

using namespace std;
using namespace tbb;
using namespace cms;

vector<string_view>
DDSpecPar::split(string_view str, const char* delims) const {
  vector<string_view> ret;
  
  string_view::size_type start = 0;
  auto pos = str.find_first_of(delims, start);
  while(pos != string_view::npos) {
    if(pos != start) {
      ret.emplace_back(str.substr(start, pos - start));
    }
    start = pos + 1;
    pos = str.find_first_of(delims, start);
  }
  if(start < str.length())
    ret.emplace_back(str.substr(start, str.length() - start));
  return ret;
}

string_view
DDSpecPar::realTopName(string_view input) const {
  string_view v = input;
  auto first = v.find_first_of("//");
  v.remove_prefix(min(first+2, v.size()));
  return v;
}

vector<string_view>
DDSpecPar::vPathsTo(unsigned int level) const {
  vector<string_view> result;
  for(auto const& i : paths) {
    vector<string_view> toks = split(i, "/");
    if(level == toks.size())
      result.emplace_back(realTopName(i));
  }
  return result;
}

vector<string_view>
DDSpecPar::tails(const vector<string_view>& fullPath) const {
  vector<string_view> result;
  for(auto const& v : fullPath) {
    auto found = v.find_last_of("/");
    if(found != v.npos) {
      result.emplace_back(v.substr(found + 1));
    }
  }
  return result;
}

string_view
DDSpecPar::strValue(const char* key) const {
  auto const& item = spars.find(key);
  if(item == end(spars))
    return string_view();
  return *begin(item->second);
}

double
DDSpecPar::dblValue(const char* key) const {
  auto const& item = numpars.find(key);
  if(item == end(numpars))
    return -1;
  return *begin(item->second);
}

void
DDSpecParRegistry::filter(DDSpecParRefMap& refToMap,
			  string_view attribute, string_view value) const {

  bool found(false);
  for_each(begin(specpars), end(specpars), [&refToMap, &attribute,
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
      refToMap.emplace(&k.first, &k.second);
    }
  });
}
