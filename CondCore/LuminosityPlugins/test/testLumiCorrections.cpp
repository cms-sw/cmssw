#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/LuminosityPlugins/plugins/LumiCorrections_PayloadInspector.cc"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
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

  // LumiCorrectionsSummary
  std::string tag = "LumiPCC_Corrections_prompt";
  cond::Time_t start = static_cast<unsigned long long>(1545372182773899);
  cond::Time_t end = static_cast<unsigned long long>(1545372182773899);

  edm::LogPrint("testLumiCorrectionsSummaryPayloadInspector")
      << "## Exercising LumiCorrectionsSummary plots " << std::endl;

  LumiCorrectionsSummary test;
  test.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testLumiCorrectionsSummaryPayloadInspector") << test.data() << std::endl;

  Py_Finalize();
}
