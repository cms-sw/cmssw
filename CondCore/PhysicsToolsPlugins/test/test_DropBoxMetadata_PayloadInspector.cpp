#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/PhysicsToolsPlugins/plugins/DropBoxMetadata_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  std::string tag = "DropBoxMetadata_v5.1_express";
  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(346361);

  edm::LogPrint("test_DropBoxMetadata_PayloadInspector") << "## test Display" << std::endl;

  DropBoxMetadata_Display histo1;
  histo1.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("test_DropBoxMetadata_PayloadInspector") << histo1.data() << std::endl;

  edm::LogPrint("test_DropBoxMetadata_PayloadInspector") << "## test Compare" << std::endl;

  DropBoxMetadata_Compare histo2;
  histo2.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("test_DropBoxMetadata_PayloadInspector") << histo2.data() << std::endl;

  Py_Finalize();
}
