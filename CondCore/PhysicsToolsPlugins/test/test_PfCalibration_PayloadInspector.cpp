#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/CondDB/interface/Time.h"

#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/PhysicsToolsPlugins/plugins/PfCalibration_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

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

  std::string tag = "PFCalibration_v10_mc";

  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(1);

  edm::LogWarning("test_PfCalibration") << "test running";

  cond::payloadInspector::PlotImage<PerformancePayloadFromTFormula, SINGLE_IOV>* histograms[] = {
      new PfCalibration<PerformanceResult::PFfa_BARREL>(),
      new PfCalibration<PerformanceResult::PFfa_ENDCAP>(),
      new PfCalibration<PerformanceResult::PFfb_BARREL>(),
      new PfCalibration<PerformanceResult::PFfb_ENDCAP>(),
      new PfCalibration<PerformanceResult::PFfc_BARREL>(),
      new PfCalibration<PerformanceResult::PFfc_ENDCAP>(),
      new PfCalibration<PerformanceResult::PFfaEta_BARRELH>(),
      new PfCalibration<PerformanceResult::PFfaEta_ENDCAPH>(),
      new PfCalibration<PerformanceResult::PFfbEta_BARRELH>(),
      new PfCalibration<PerformanceResult::PFfbEta_ENDCAPH>(),
      new PfCalibration<PerformanceResult::PFfaEta_BARRELEH>(),
      new PfCalibration<PerformanceResult::PFfaEta_ENDCAPEH>(),
      new PfCalibration<PerformanceResult::PFfbEta_BARRELEH>(),
      new PfCalibration<PerformanceResult::PFfbEta_ENDCAPEH>(),
      new PfCalibration<PerformanceResult::PFfaEta_BARREL>(),
      new PfCalibration<PerformanceResult::PFfaEta_ENDCAP>(),
      new PfCalibration<PerformanceResult::PFfbEta_BARREL>(),
      new PfCalibration<PerformanceResult::PFfbEta_ENDCAP>(),
      new PfCalibration<PerformanceResult::PFfcEta_BARRELH>(),
      new PfCalibration<PerformanceResult::PFfcEta_ENDCAPH>(),
      new PfCalibration<PerformanceResult::PFfdEta_ENDCAPH>(),
      new PfCalibration<PerformanceResult::PFfcEta_BARRELEH>(),
      new PfCalibration<PerformanceResult::PFfcEta_ENDCAPEH>(),
      new PfCalibration<PerformanceResult::PFfdEta_ENDCAPEH>()};

  for (auto hist : histograms) {
    hist->process(connectionString, PI::mk_input(tag, start, end));
    std::cout << hist->data() << std::endl;
  }

  Py_Finalize();
}