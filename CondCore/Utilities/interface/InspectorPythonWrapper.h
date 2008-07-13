#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "CondCore/DBCommon/interface/ClassID.h"


using namespace boost::python;



#define PYTHON_WRAPPER(_class,_name) \
namespace { typedef cond::PayLoadInspector< _class > PythonWrapper;} \
BOOST_PYTHON_MODULE(plugin ## _name ## PyInterface) { \
  class_<PythonWrapper>("Object",init<>()) \
    .def(init<cond::IOVElement>()) \
    .def("print",&PythonWrapper::print) \
    .def("summary",&PythonWrapper::summary); \
} \
namespace { const char * pluginName_="plugin"  #_name "PyInterface"; }\
PYTHON_ID(PythonWrapper::Class, pluginName_)


