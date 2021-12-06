#include <pybind11/pybind11.h>
namespace py = pybind11;

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)

#define PAYLOAD_INSPECTOR_MODULE(PAYLOAD_TYPENAME) PYBIND11_MODULE(plugin##PAYLOAD_TYPENAME##_PayloadInspector, m)

#define PAYLOAD_INSPECTOR_CLASS(CLASS_NAME)                                   \
  py::class_<CLASS_NAME, cond::payloadInspector::PlotBase>(                   \
      m,                                                                      \
      STRINGIZE(PPCAT(plot_, CLASS_NAME)))                                    \
          .def(py::init<>())                                                  \
          .def("process", &cond::payloadInspector::PlotBase::process)         \
          .def("payloadType", &cond::payloadInspector::PlotBase::payloadType) \
          .def("title", &cond::payloadInspector::PlotBase::title)             \
          .def("isSingleIov", &cond::payloadInspector::PlotBase::isSingleIov) \
          .def("isTwoTags", &cond::payloadInspector::PlotBase::isTwoTags)     \
          .def("data", &cond::payloadInspector::PlotBase::data);
