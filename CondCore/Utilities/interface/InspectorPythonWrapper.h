#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "CondCore/DBCommon/interface/ClassID.h"


using namespace boost::python;

namespace {
  template<typename Wrapper>
  void define() {
    typedef typename Wrapper::Extractor Extractor;
    class_<Extractor>("Extractor", init<>())
      .def(init<std::string, std::vector<int> )
      .def("values",&Extractor::values, return_value_policy<copy_const_reference>())
      ;

    class_<Wrapper>("Object",init<>()) 
      .def(init<cond::IOVElement>())
      .def("print",&Wrapper::print)
      .def("summary",&Wrapper::summary) 
      .def("extract",&Wrapper::extractor)
      ; 
  }
}


#define PYTHON_WRAPPER(_class,_name) \
namespace { typedef cond::PayLoadInspector< _class > PythonWrapper;} \
 BOOST_PYTHON_MODULE(plugin ## _name ## PyInterface) { define<PythonWrapper>(); } 
namespace { const char * pluginName_="plugin"  #_name "PyInterface"; }\
PYTHON_ID(PythonWrapper::Class, pluginName_)


