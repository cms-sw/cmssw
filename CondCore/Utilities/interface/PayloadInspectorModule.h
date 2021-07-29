 #include <pybind11/pybind11.h>

namespace py = pybind11;

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)

#define PAYLOAD_INSPECTOR_MODULE(PAYLOAD_TYPENAME) PYBIND11_MODULE(plugin##PAYLOAD_TYPENAME##_PayloadInspector,m)

#define PAYLOAD_INSPECTOR_CLASS(CLASS_NAME)                                                                    \
  py::class_<CLASS_NAME,cond::payloadInspector::PlotBase>(m, STRINGIZE(PPCAT(plot_, CLASS_NAME)))              \
          .def(py::init<>())                                                                                   \
          .def("process", &CLASS_NAME::process)                                                                \
          .def("payloadType", &CLASS_NAME::payloadType)                                                        \
          .def("title", &CLASS_NAME::title)                                                                    \
          .def("isSingleIov", &CLASS_NAME::isSingleIov)                                                        \
          .def("isTwoTags", &CLASS_NAME::isTwoTags)                                                            \
          .def("data", &CLASS_NAME::data);
