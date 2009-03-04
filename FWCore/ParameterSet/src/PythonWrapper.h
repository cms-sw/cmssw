#ifndef ParameterSet_PythonWrapper_h
#define ParameterSet_PythonWrapper_h

#include <vector>
#include <string>
#include "FWCore/ParameterSet/interface/BoostPython.h"
using namespace boost::python;

namespace edm {
void
pythonToCppException(const std::string& iType);

//  boost::python::list toPythonList(const std::vector<std::string> & v);
  // utility to translate from an STL vector of strings to
  // a Python list
  template<typename T>
  boost::python::list toPythonList(const std::vector<T> & v) {
    boost::python::list result;
    for(unsigned i = 0; i < v.size(); ++i) {
       result.append(v[i]);
    }
    return result;
  }



  // and back.  Destroys the input via pop()s
  template<typename T>
  std::vector<T> toVector(boost::python::list & l)
  {
    std::vector<T> result;
    unsigned n = PyList_Size(l.ptr());
    object iter_obj(handle<>(PyObject_GetIter(l.ptr())));
    for(unsigned i = 0; i < n; ++i)
    {
      object obj = extract<object>(iter_obj.attr("next")());
      result.push_back(extract<T>(obj)); 
    }
    return result;
  }
}

#endif

