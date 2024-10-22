#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "CondCore/CTPPSPlugins/interface/PPSTimingCalibrationPayloadInspectorHelper.h"

int main(int argc, char** argv) {
  Py_Initialize();

  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  std::string tag = "CTPPPSTimingCalibration_HPTDC_byPCL_v0_prompt";
  cond::Time_t start = static_cast<unsigned long long>(355892);
  cond::Time_t end = static_cast<unsigned long long>(357079);

  edm::LogPrint("testPPSCalibrationPI") << "## Exercising TimingCalibration plots ";

  ParametersPerChannel<PPSTimingCalibrationPI::parameter0, PPSTimingCalibration> test;
  py::dict inputs;
  inputs["db (0,1)"] = "0";
  inputs["station (1,2)"] = "1";
  inputs["plane (0-3)"] = "0";
  test.setInputParamValues(inputs);
  test.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testparametersPerChannel") << test.data();

  inputs.clear();
#if PY_MAJOR_VERSION >= 3
  Py_INCREF(inputs.ptr());
#endif

  Py_Finalize();
}
