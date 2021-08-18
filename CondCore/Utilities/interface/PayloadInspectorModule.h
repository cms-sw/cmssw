#include <boost/python.hpp>

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)

#define PAYLOAD_INSPECTOR_MODULE(PAYLOAD_TYPENAME) BOOST_PYTHON_MODULE(plugin##PAYLOAD_TYPENAME##_PayloadInspector)

#define PAYLOAD_INSPECTOR_CLASS(CLASS_NAME)                                                   \
  boost::python::class_<CLASS_NAME, boost::python::bases<cond::payloadInspector::PlotBase> >( \
      STRINGIZE(PPCAT(plot_, CLASS_NAME)), boost::python::init<>())                           \
          .def("process", &cond::payloadInspector::PlotBase::process)                         \
          .def("payloadType", &cond::payloadInspector::PlotBase::payloadType)                 \
          .def("title", &cond::payloadInspector::PlotBase::title)                             \
          .def("isSingleIov", &cond::payloadInspector::PlotBase::isSingleIov)                 \
          .def("isTwoTags", &cond::payloadInspector::PlotBase::isTwoTags)                     \
          .def("data", &cond::payloadInspector::PlotBase::data);
