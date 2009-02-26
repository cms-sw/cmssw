#ifndef ParameterSet_PythonWrapper_h
#define ParameterSet_PythonWrapper_h

#include <vector>
#include <string>
#include "FWCore/ParameterSet/interface/BoostPython.h"
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
    try {
      boost::python::object iter_obj 
        = boost::python::object(boost::python::handle<>(PyObject_GetIter(l.ptr())));
      while(1) {
	boost::python::object obj = boost::python::extract<boost::python::object>(iter_obj.attr("next")());
	result.push_back(boost::python::extract<T>(obj)); 
      }
    }
    catch(boost::python::error_already_set) {
      // This is how it finds the end of the list.
      //  By throwing an exception.
      PyErr_Clear(); 
    }

    return result;
  }
}

#endif

