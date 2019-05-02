#ifndef FWCore_PyDevParameterSet_PyBind11Wrapper_h
#define FWCore_PyDevParameterSet_PyBind11Wrapper_h

#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace cmspython3 {
  void pythonToCppException(const std::string& iType, const std::string& error);

  // utility to translate from an STL vector of strings to
  // a Python list

  template <typename T>
  pybind11::list toPython11List(const std::vector<T>& v) {
    pybind11::list result = pybind11::cast(v);
    return result;
  }

  // and back.  Destroys the input via pop()s - well probably not
  template <typename T>
  std::vector<T> toVector(pybind11::list& l) {
    std::vector<T> result;
    result.reserve(l.size());
    for (unsigned i = 0; i < l.size(); ++i) {
      result.push_back(l[i].cast<T>());
    }
    return result;
  }
}  // namespace cmspython3

#endif
