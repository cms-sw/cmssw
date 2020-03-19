#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/plugins/BasicP_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Not enough arguments given." << std::endl;
    return 0;
  }

  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("oracle://cms_orcon_adg/CMS_CONDITIONS");

  std::string tag = std::string(argv[1]);
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t since = boost::lexical_cast<unsigned long long>(argv[2]);

  std::cout << "## PNG Histo" << std::endl;

  BasicPayload_data6 histo1;
  histo1.process(connectionString, tag, runTimeType, since, since);
  std::cout << histo1.data() << std::endl;
}
