#ifndef ParameterSet_PythonWrapper_h
#define ParameterSet_PythonWrapper_h

#include <vector>
#include <string>
#include <boost/python.hpp>
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
namespace edm {
//  boost::python::list toPythonList(const std::vector<std::string> & v);
  // utility to translate from an STL vector of strings to
  // a Python list
static
void
pythonToCppException(const std::string& iType)
{
  using namespace boost::python;
  PyObject *exc=NULL, *val=NULL, *trace=NULL;
  PyErr_Fetch(&exc,&val,&trace);
  PyErr_NormalizeException(&exc,&val,&trace);
  handle<> hExc(allow_null(exc));
  handle<> hVal(allow_null(val));
  handle<> hTrace(allow_null(trace));
  if(hTrace) {
    object oTrace(hTrace);
  }

  if(hVal && hExc) {
    object oExc(hExc);
    object oVal(hVal);
    handle<> hStringVal(PyObject_Str(oVal.ptr()));
    object stringVal( hStringVal );

    handle<> hStringExc(PyObject_Str(oExc.ptr()));
    object stringExc( hStringExc);

    //PyErr_Print();
    throw cms::Exception(iType) <<"python encountered the error: "
				<< PyString_AsString(stringExc.ptr())<<" "
				<< PyString_AsString(stringVal.ptr())<<"\n";
  } else {
    throw cms::Exception(iType)<<" unknown python problem occurred.\n";
  }
}


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

