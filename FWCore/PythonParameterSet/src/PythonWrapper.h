#ifndef FWCore_PythonParameterSet_PythonWrapper_h
#define FWCore_PythonParameterSet_PythonWrapper_h

#include <vector>
#include <string>
#include "FWCore/PythonParameterSet/interface/BoostPython.h"

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
    boost::python::object iter_obj(boost::python::handle<>(PyObject_GetIter(l.ptr())));
    for(unsigned i = 0; i < n; ++i)
    {
      boost::python::object obj = boost::python::extract<boost::python::object>(iter_obj.attr("next")());
      result.push_back(boost::python::extract<T>(obj)); 
    }
    return result;
  }
}

#endif
