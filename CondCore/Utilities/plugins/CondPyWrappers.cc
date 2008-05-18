// python wrappers for CondDB
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/Utilities/interface/CondPyInterface.h"


#include <boost/python.hpp>

using namespace boost::python;

namespace {
  exceptionTranslator(const edm::Exception & e)
  {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  exceptionTranslator(const std::exception & e)
  {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
}

BOOST_PYTHON_MODULE(pluginCondDBPyInterface) {

  class_<cond::FWIncantation>("FWIncantation", init<>());

  class_<cond::CondDB>("CondDB", init<>())
    .def("listTags", &cond::CondDB::allTags);
  
  class_<cond::RDBMS>("RDBMS", init<>())
    .def(init<std::string>())
    .def(init<std::string, std::string>())
    .def("getDB", &cond::RDBMS::getDB);

  register_exception_translator<edm::Exception>(exceptionTranslator);
  register_exception_translator<std::exception>(exceptionTranslator);


}
