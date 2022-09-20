#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondCore/CTPPSPlugins/interface/CTPPSRPAlignmentCorrectionsDataHelper.h"

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

  std::string tag = "CTPPSRPAlignment_real_offline_v8";
  cond::Time_t start = static_cast<unsigned long long>(273725);
  cond::Time_t end = static_cast<unsigned long long>(325159);

  edm::LogPrint("testPPSCalibrationPI") << "## Exercising PPSRPAlignmentCorrections plots ";

  RPShift_History<CTPPSRPAlignment::RP::RP3, CTPPSRPAlignment::Shift::x, false, CTPPSRPAlignmentCorrectionsData> test;
  test.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testRPShift_History") << test.data();
  Py_Finalize();
}
