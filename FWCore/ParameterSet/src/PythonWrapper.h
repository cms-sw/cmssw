#ifndef ParameterSet_PythonWrapper_h
#define ParameterSet_PythonWrapper_h

#include <vector>
#include <string>
#include <boost/python.hpp>

// bad form?
using namespace boost::python;

namespace edm {
//  boost::python::list toPythonList(const std::vector<std::string> & v);
  // utility to translate from an STL vector of strings to
  // a Python list
  template<class T>
  boost::python::list toPythonList(const std::vector<T> & v)
  {
    boost::python::list result;
    for(unsigned i = 0; i < v.size(); ++i)
    {
       result.append(v[i]);
    }
    return result;
  }



  // and back.  Destroys the input via pop()s
  template<class T>
  std::vector<T> toVector(boost::python::list & l)
  {
    std::vector<T> result;
    try
    {
      object iter_obj = object( handle<>( PyObject_GetIter( l.ptr() ) ));

      while( 1 )
	    {
	      object obj = extract<object>( iter_obj.attr( "next" )() );
	      result.push_back(extract<T>( obj )); 
	    }
    }
    catch( error_already_set )
    {
      PyErr_Clear(); 
    }

  return result;
  }

}

#endif

