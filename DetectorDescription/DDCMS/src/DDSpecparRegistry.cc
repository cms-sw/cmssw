#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/Detector.h"
#include <algorithm>

using namespace std;
using namespace cms;
using namespace edm;

string_view DDSpecPar::strValue(const string& key) const {
  auto const& item = spars.find(key);
  if (item == end(spars))
    return string();
  return *begin(item->second);
}

bool DDSpecPar::hasValue(const string& key) const {
  if (numpars.find(key) != end(numpars))
    return true;
  else
    return false;
}

template <>
std::vector<double> DDSpecPar::value<std::vector<double>>(const string& key) const {
  std::vector<double> result;

  auto const& nitem = numpars.find(key);
  if (nitem != end(numpars)) {
    return std::vector<double>(begin(nitem->second), end(nitem->second));
  }

  auto const& sitem = spars.find(key);
  if (sitem != end(spars)) {
    std::transform(begin(sitem->second), end(sitem->second), std::back_inserter(result), [](auto& i) -> double {
      return dd4hep::_toDouble(i);
    });
  }

  return result;
}

template <>
std::vector<std::string> DDSpecPar::value<std::vector<std::string>>(const string& key) const {
  std::vector<std::string> result;

  auto const& nitem = numpars.find(key);
  if (nitem != end(numpars)) {
    std::transform(begin(nitem->second), end(nitem->second), std::back_inserter(result), [](auto& i) -> std::string {
      return std::to_string(i);
    });

    return result;
  }

  auto const& sitem = spars.find(key);
  if (sitem != end(spars)) {
    return std::vector<std::string>(begin(sitem->second), end(sitem->second));
  }

  return result;
}

double DDSpecPar::dblValue(const string& key) const {
  auto const& item = numpars.find(key);
  if (item == end(numpars))
    return 0;
  return *begin(item->second);
}

void DDSpecParRegistry::filter(DDSpecParRefs& refs, string_view attribute, string_view value) const {
  bool found(false);
  for_each(begin(specpars), end(specpars), [&refs, &attribute, &value, &found](const auto& k) {
    found = false;
    for_each(begin(k.second.spars), end(k.second.spars), [&](const auto& l) {
      if (l.first == attribute) {
        for_each(begin(l.second), end(l.second), [&](const auto& m) {
          if (m == value)
            found = true;
        });
      }
    });
    if (found) {
      refs.emplace_back(&k.second);
    }
  });
}

void DDSpecParRegistry::filter(DDSpecParRefs& refs, string_view attribute) const {
  bool found(false);
  for_each(begin(specpars), end(specpars), [&refs, &attribute, &found](const auto& k) {
    found = false;
    for_each(begin(k.second.spars), end(k.second.spars), [&](const auto& l) {
      if (l.first == attribute) {
        found = true;
      }
    });
    if (found) {
      refs.emplace_back(&k.second);
    }
  });
}

std::vector<std::string_view> DDSpecParRegistry::names() const {
  std::vector<std::string_view> result;
  for_each(begin(specpars), end(specpars), [&result](const auto& i) { result.emplace_back(i.first); });
  return result;
}

bool DDSpecParRegistry::hasSpecPar(std::string_view name) const {
  auto const& result =
      find_if(begin(specpars), end(specpars), [&name](const auto& i) { return (i.first.compare(name) == 0); });
  if (result != end(specpars))
    return true;
  else
    return false;
}

const DDSpecPar* DDSpecParRegistry::specPar(std::string_view name) const {
  auto const& result =
      find_if(begin(specpars), end(specpars), [&name](const auto& i) { return (i.first.compare(name) == 0); });
  if (result != end(specpars)) {
    return &result->second;
  } else {
    return nullptr;
  }
}
