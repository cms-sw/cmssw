#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <DD4hep/Filter.h>

#include <cmath>
#include "tbb/concurrent_vector.h"

template <>
std::vector<int> cms::DDCompactView::getVector<int>(const std::string& key) const {
  std::vector<int> result;
  const auto& vmap = this->detector()->vectors();
  for (auto const& it : vmap) {
    if (dd4hep::dd::noNamespace(it.first) == key) {
      std::transform(
          it.second.begin(), it.second.end(), std::back_inserter(result), [](int n) -> int { return (int)n; });
      return result;
    }
  }
  return result;
}

template <>
std::vector<double> cms::DDCompactView::getVector<double>(const std::string& key) const {
  const auto& vmap = this->detector()->vectors();
  for (auto const& it : vmap) {
    if (dd4hep::dd::noNamespace(it.first) == key) {
      return it.second;
    }
  }
  return std::vector<double>();
}

template <>
std::vector<double> const& cms::DDCompactView::get<std::vector<double>>(const std::string& key) const {
  const auto& vmap = this->detector()->vectors();
  for (auto const& it : vmap) {
    if (dd4hep::dd::noNamespace(it.first) == key) {
      return it.second;
    }
  }
  throw cms::Exception("DDError") << "no vector<double> with name " << key;
}

template <>
tbb::concurrent_vector<double> const& cms::DDCompactView::get<tbb::concurrent_vector<double>>(
    const std::string& name, const std::string& key) const {
  const auto& spec = specpars().specPar(name);
  if (spec != nullptr) {
    auto const& nitem = spec->numpars.find(key);
    if (nitem != end(spec->numpars)) {
      return nitem->second;
    }
  }
  throw cms::Exception("DDError") << "no SpecPar with name " << name << " and vector<double> key " << key;
}
