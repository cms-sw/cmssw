#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/PCLConfigPlugins/plugins/AlignPCLThresholds_PayloadInspector.cc"
#include "CondCore/PCLConfigPlugins/plugins/AlignPCLThresholdsHG_PayloadInspector.cc"

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

  // AlignPCLThresholds
  std::string tag = "SiPixelAliThresholds_offline_v0";
  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(359659);

  edm::LogPrint("testAlignPCLThresholdsPayloadInspector") << "## Exercising AlignPCLThresholds plots " << std::endl;

  AlignPCLThresholds_Display histo1;
  histo1.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testAlignPCLThresholdsPayloadInspector") << histo1.data() << std::endl;

  std::string tag2 = "SiPixelAliThresholds_express_v0";

  AlignPCLThresholds_CompareTwoTags histo2;
  histo2.process(connectionString, PI::mk_input(tag, start, start, tag2, start, start));
  edm::LogPrint("testAlignPCLThresholdsPayloadInspector") << histo2.data() << std::endl;

  // AlignPCLThresholdsHG
  tag = "SiPixelAliThresholdsHG_express_v0";
  edm::LogPrint("testAlignPCLThresholdsPayloadInspector") << "## Exercising AlignPCLThresholdsHG plots " << std::endl;

  AlignPCLThresholdsHG_Display histo3;
  histo3.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testAlignPCLThresholdsPayloadInspector") << histo3.data() << std::endl;

  AlignPCLThresholdsHG_Compare histo4;
  histo4.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testAlignPCLThresholdsPayloadInspector") << histo4.data() << std::endl;

  Py_Finalize();
}
