#ifndef FWCore_ParameterSet_getFixedSizeArray_h
#define FWCore_ParameterSet_getFixedSizeArray_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace edm {

  template<class T, std::size_t N>
  std::array<T, N>
  getFixedSizeArray(ParameterSet const& pset, std::string const& name) {
    std::vector<T> vec = pset.getParameter<std::vector<T>>(name);
    if (vec.size() != N) {
      throw Exception(errors::Configuration)
        << "The parameter '" << name
        << "' should have " << N << " elements, but has " << vec.size()
        << " elements in the configuration.\n";
    }
    std::array<T, N> a {};
    std::copy_n(vec.begin(), N, a.begin());
    return a;
  }
}
#endif
