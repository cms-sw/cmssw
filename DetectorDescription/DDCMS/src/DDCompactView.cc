#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"

#include <cmath>

template <>
std::vector<int> cms::DDCompactView::getVector<int>(const std::string& key) const {
  const cms::DDVectorsMap& vmap = this->detector()->vectors();
  std::vector<int> result;
  for (auto const& it : vmap) {
    if (cms::dd::noNamespace(it.first) == key) {
      for (const auto& i : it.second) {
        result.emplace_back(std::round(i));
      }
      break;
    }
  }
  return result;
}

template <>
std::vector<double> cms::DDCompactView::getVector<double>(const std::string& key) const {
  const cms::DDVectorsMap& vmap = this->detector()->vectors();
  std::vector<double> result;

  for (auto const& it : vmap) {
    if (cms::dd::noNamespace(it.first) == key) {
      for (const auto& i : it.second) {
        result.emplace_back(i);
      }
      break;
    }
  }
  return result;
}
