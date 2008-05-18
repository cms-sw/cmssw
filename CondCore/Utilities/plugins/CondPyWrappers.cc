// python wrappers for CondDB
#include "CondCore/Utilities/interface/CondPyInterface.h"


#include <boost/python.hpp>

using namespace boost::python;



BOOST_PYTHON_MODULE(pluginCondDBPyInterface) {

  class_<cond::FWIncantation>("FWIncantation", init<>());

  class_<cond::CondDB>("CondDB", init<>())
    .def("listTags", &cond::CondDB::allTags);
  
  class_<cond::RDBMS>("RDBMS", init<>())
    .def(init<std::string>())
    .def(init<std::string, std::string>())
    .def("getDB", &cond::CondDB::getDB);


}




















}
