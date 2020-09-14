#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

int run(const std::string& connectionString) {
  try {
    //*************
    std::cout << "> Connecting with db in " << connectionString << std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    connPool.configure();
    Session session = connPool.createSession(connectionString);
    session.transaction().start();
    cond::RunInfo_t r = session.getLastRun();
    std::cout << "Last run: " << r.run << " start:" << r.start << std::endl;
    if (r.isOnGoing())
      std::cout << "Run is ongoing" << std::endl;
    else
      std::cout << "Run was ending on " << r.end << std::endl;
    session.transaction().commit();
  } catch (cond::Exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return 1;
  }
  //std::cout << "## RunInfo test successfully completed." << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  const edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
  const edm::ServiceRegistry::Operate operate(services);
  int ret = 0;
  std::string connectionString0("frontier://FrontierProd/CMS_CONDITIONS");
  //std::string connectionString0("frontier://FrontierProd/CMS_CONDITIONS");
  //std::string connectionString0("oracle://cms_orcoff_prep/CMS_CONDITIONS");
  ret = run(connectionString0);
  return ret;
}
