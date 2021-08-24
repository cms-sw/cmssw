#include "CondCore/Utilities/interface/PayloadInspector.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(pluginModule_PayloadInspector, m) {
  py::class_<cond::payloadInspector::ModuleVersion>(m, "ModuleVersion")
      .def_readonly_static("label", &cond::payloadInspector::ModuleVersion::label);
  py::class_<cond::payloadInspector::PlotBase>(m, "PlotBase")
      .def("process", &cond::payloadInspector::PlotBase::process)
      .def("payloadType", &cond::payloadInspector::PlotBase::payloadType)
      .def("type", &cond::payloadInspector::PlotBase::type)
      .def("title", &cond::payloadInspector::PlotBase::title)
      .def("isSingleIov", &cond::payloadInspector::PlotBase::isSingleIov)
      .def("isTwoTags", &cond::payloadInspector::PlotBase::isTwoTags)
      .def("inputParams", &cond::payloadInspector::PlotBase::inputParams)
      .def("setInputParamValues", &cond::payloadInspector::PlotBase::setInputParamValues)
      .def("data", &cond::payloadInspector::PlotBase::data);
}
