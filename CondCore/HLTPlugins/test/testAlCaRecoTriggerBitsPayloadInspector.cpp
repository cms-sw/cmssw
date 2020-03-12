#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/HLTPlugins/plugins/AlCaRecoTriggerBits_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  std::string tag = "AlCaRecoHLTpaths8e29_1e31_v7_hlt";
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t start = boost::lexical_cast<unsigned long long>(270000);
  cond::Time_t end = boost::lexical_cast<unsigned long long>(304820);

  std::cout << "## AlCaRecoTriggerBit Histos" << std::endl;

  AlCaRecoTriggerBits_Display histo1;
  histo1.process(connectionString, tag, runTimeType, 1, 1);
  std::cout << histo1.data() << std::endl;

  AlCaRecoTriggerBits_Compare histo2;
  histo2.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo2.data() << std::endl;
}
