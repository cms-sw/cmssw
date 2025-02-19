#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "CondCore/DBCommon/interface/ClassID.h"


namespace condPython {
  template<typename T>
  void defineWhat() {
    using namespace boost::python;
    typedef cond::ExtractWhat<T> What;
    class_<What>("What",init<>());
  }

}

namespace {
  
  template<typename Wrapper>
  void define() {
    using namespace boost::python;
    typedef typename Wrapper::Extractor Extractor;
    typedef typename Extractor::What What;

    condPython::defineWhat<typename Extractor::Class>();

    class_<Extractor>("Extractor", init<>())
      .def(init<What>())
      .def("what",Extractor::what)
      .def("values",&Extractor::values, return_value_policy<copy_const_reference>())
      ;

    class_<Wrapper>("Object",init<>()) 
      .def(init<cond::CondDB>())
      .def("load",&Wrapper::load)
      .def("dump",&Wrapper::dump)
      .def("dumpXML",&Wrapper::dumpXML)
      .def("plot",&Wrapper::plot)
      .def("summary",&Wrapper::summary) 
      .def("extract",&Wrapper::extract)
      .def("trend_plot",&Wrapper::trend_plot)
      .def("summary_adv",&Wrapper::summary_adv)
      .def("dumpFile", &Wrapper::dumpFile)
      ; 
  }
}


#define PYTHON_WRAPPER(_class,_name) \
namespace { typedef cond::PayLoadInspector< _class > PythonWrapper;} \
 BOOST_PYTHON_MODULE(plugin ## _name ## PyInterface) { define<PythonWrapper>(); } \
namespace { const char * pluginName_="plugin"  #_name "PyInterface"; }\
 PYTHON_ID(PythonWrapper::Class, pluginName_, _name)


