#include "CondCore/Utilities/interface/PayloadInspector.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

BOOST_PYTHON_MODULE(pluginModule_PayloadInspector) {
  boost::python::class_<cond::payloadInspector::ModuleVersion>("ModuleVersion")
      .def_readonly("label", &cond::payloadInspector::ModuleVersion::label);
  boost::python::class_<cond::payloadInspector::PlotBase>("PlotBase")
      .def("process", &cond::payloadInspector::PlotBase::process)
      .def("processTwoTags", &cond::payloadInspector::PlotBase::processTwoTags)
      .def("payloadType", &cond::payloadInspector::PlotBase::payloadType)
      .def("type", &cond::payloadInspector::PlotBase::type)
      .def("title", &cond::payloadInspector::PlotBase::title)
      .def("isSingleIov", &cond::payloadInspector::PlotBase::isSingleIov)
      .def("isTwoTags", &cond::payloadInspector::PlotBase::isTwoTags)
      .def("inputParams", &cond::payloadInspector::PlotBase::inputParams)
      .def("setInputParamValues", &cond::payloadInspector::PlotBase::setInputParamValues)
      .def("data", &cond::payloadInspector::PlotBase::data);
}
