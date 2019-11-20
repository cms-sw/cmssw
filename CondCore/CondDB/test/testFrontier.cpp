#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include "MyTestData.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  const edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
  const edm::ServiceRegistry::Operate operate(services);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");
  std::cout << "# Connecting with db in " << connectionString << std::endl;
  try {
    //*************
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    Session session = connPool.createSession(connectionString);
    session.transaction().start();
    IOVProxy iov = session.readIov("runinfo_31X_hlt", true);
    std::cout << "Loaded size=" << iov.loadedSize() << std::endl;
    session.transaction().commit();
  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}
