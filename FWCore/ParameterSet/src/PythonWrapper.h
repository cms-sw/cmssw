#ifndef ParameterSet_PythonWrapper_h
#define ParameterSet_PythonWrapper_h

#include <vector>
#include <string>
#include <boost/python.hpp>

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
    bool is_ok = true;
    try
    {
      while( is_ok ) {
        boost::python::extract<T>  x( l.pop( 0 ));

        if( x.check()) {
          result.push_back( x());
        } else {
          is_ok = false;
        }
      }
    }
    // the pop will end in an exception
    catch(...)
    {
    }

    return result;
  }

}

#endif

