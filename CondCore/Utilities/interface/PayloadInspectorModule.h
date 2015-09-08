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

#define PAYLOAD_INSPECTOR_CLASS( CLASS_NAME ) using namespace boost::python; \
  class_< CLASS_NAME >( STRINGIZE(PPCAT(plot_,CLASS_NAME)), init<>()) \
  .def("objectType",&CLASS_NAME::objectType ) \
  .def("title",&CLASS_NAME::title ) \
  .def("info",&CLASS_NAME::info ) \
  .def("data",&CLASS_NAME::data ) \
  ;

#define PAYLOAD_INSPECTOR_FUNCTION( FUNCTION_NAME ) using namespace boost::python; \
  def (STRINGIZE(PPCAT(plot_,FUNCTION_NAME)), FUNCTION_NAME)
