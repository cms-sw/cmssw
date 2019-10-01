#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"

#include <cmath>

template <>
std::vector<int> cms::DDCompactView::getVector<int>(std::string_view key) const {
  cms::DDVectorsMap vmap = this->detector()->vectors();
  std::vector<int> temp;
  for (auto const& it : vmap) {
    if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), key)) {
      for (const auto& i : it.second) {
        temp.emplace_back(std::round(i));
      }
    }
  }
  return temp;
}

template <>
std::vector<double> cms::DDCompactView::getVector<double>(std::string_view key) const {
  cms::DDVectorsMap vmap = this->detector()->vectors();
  std::vector<double> temp;
  for (auto const& it : vmap) {
    if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), key)) {
      for (const auto& i : it.second) {
        temp.emplace_back(i);
      }
    }
  }
  return temp;
}
