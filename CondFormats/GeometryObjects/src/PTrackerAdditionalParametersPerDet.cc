#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"
#include <iostream>

namespace {
  template <typename T>
  const T getThisParam(const std::vector<std::vector<T>>& params, size_t index_outer, size_t index_inner) {
    if (index_outer >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index_outer) + " is out of range.");
    if (index_inner >= params[index_outer].size())
      throw std::out_of_range("Parameter with index " + std::to_string(index_inner) + " is out of range.");
    return params[index_outer][index_inner];
  }

  template <typename T>
  void setThisParam(std::vector<std::vector<T>>& params, size_t index_outer, const T& value) {
    if (index_outer >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index_outer) + " is out of range.");
    params[index_outer].push_back(value);
  }

  template <typename T>
  const std::vector<T>& getAllParams(const std::vector<std::vector<T>>& params, size_t index_outer) {
    if (index_outer >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index_outer) + " is out of range.");
    return params[index_outer];
  }

}  // namespace

int PTrackerAdditionalParametersPerDet::getGeographicalId(int theIndex) const {
  return getThisParam(intParams_, GEOGRAPHICAL_ID, theIndex);
}

std::vector<int> PTrackerAdditionalParametersPerDet::getAllGeographicalIds() const {
  return getAllParams(intParams_, GEOGRAPHICAL_ID);
}

void PTrackerAdditionalParametersPerDet::setGeographicalId(int geographicalId) {
  setThisParam(intParams_, GEOGRAPHICAL_ID, geographicalId);
}

//This doesn't work properly because intParams_ and boolParams_ are vectors of vecotrs - the outer vector should be the number of parameters and the inner vector the number of geometricDets.
