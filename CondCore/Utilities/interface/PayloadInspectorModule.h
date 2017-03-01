#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)

#define PAYLOAD_INSPECTOR_MODULE( PAYLOAD_TYPENAME ) BOOST_PYTHON_MODULE( plugin ## PAYLOAD_TYPENAME ## _PayloadInspector )

#define PAYLOAD_INSPECTOR_CLASS( CLASS_NAME ) \
boost::python::class_< CLASS_NAME >( STRINGIZE(PPCAT(boost::python::plot_,CLASS_NAME)), boost::python::init<>()) \
  .def("objectType",&CLASS_NAME::objectType ) \
  .def("title",&CLASS_NAME::title ) \
  .def("info",&CLASS_NAME::info ) \
  .def("data",&CLASS_NAME::data ) \
  ;

#define PAYLOAD_INSPECTOR_FUNCTION( FUNCTION_NAME ) \
def (STRINGIZE(PPCAT(boost::python::plot_,FUNCTION_NAME)), FUNCTION_NAME)
