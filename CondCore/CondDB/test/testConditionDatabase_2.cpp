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

int readIov(IOVProxy& proxy, cond::Time_t targetTime, bool expectedOk) {
  bool found = false;
  cond::Iov_t iov;
  std::cout << "#Testing tag " << proxy.tagInfo().name << std::endl;
  try {
    iov = proxy.getInterval(targetTime);
    if (iov.since < cond::time::MAX_VAL)
      found = true;
  } catch (const Exception& e) {
    std::cout << "# IOV " << targetTime << " not found: " << e.what() << std::endl;
  }
  if (expectedOk) {
    if (found) {
      std::cout << "#OK: found iov with since " << iov.since << " - till " << iov.till << " for time " << targetTime
                << std::endl;
      return 0;
    } else {
      std::cout << "#OK: no valid iov found for time " << targetTime << std::endl;
      return -1;
    }
  } else {
    if (found) {
      std::cout << "#ERROR: found iov=" << iov.since << " for time " << targetTime << std::endl;
      return -1;
    } else {
      std::cout << "#OK: no valid iov found for time " << targetTime << std::endl;
      return 0;
    }
  }
}

int run(const std::string& connectionString) {
  try {
    //*************
    std::cout << "> Connecting with db in " << connectionString << std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    Session session = connPool.createSession(connectionString, true);
    session.transaction().start(false);
    std::string pay0("Payload #0");
    std::string pay1("Payload #1");
    auto p0 = session.storePayload(pay0);
    auto p1 = session.storePayload(pay1);
    IOVEditor editor;
    if (!session.existsIov("MyTag")) {
      editor = session.createIov<std::string>("MyTag", cond::runnumber);
      editor.setDescription("Test for timestamp selection");
      editor.insert(100, p0);
      editor.insert(200, p1);
      editor.insert(1001, p0);
      editor.insert(1500, p1);
      editor.insert(2100, p0);
      editor.insert(2500, p1);
      editor.insert(10000, p0);
      std::cout << "> inserted 7 iovs..." << std::endl;
      editor.flush();
      std::cout << "> iov changes flushed..." << std::endl;
    }
    if (!session.existsIov("MyTag2")) {
      editor = session.createIov<std::string>("MyTag2", cond::runnumber);
      editor.setDescription("Test for timestamp selection");
      editor.insert(100, p0);
      std::cout << "> inserted 1 iovs..." << std::endl;
      editor.flush();
      std::cout << "> iov changes flushed..." << std::endl;
    }
    session.transaction().commit();
    std::cout << "> iov changes committed!..." << std::endl;
    ::sleep(2);
    session.transaction().start();
    IOVProxy proxy = session.readIov("MyTag");
    readIov(proxy, 1, false);
    readIov(proxy, 100, true);
    readIov(proxy, 1499, true);
    readIov(proxy, 1500, true);
    readIov(proxy, 20000, true);
    IOVArray iovs = proxy.selectAll();
    iovs.find(101);
    for (const auto i : iovs) {
      std::cout << "# iov since " << i.since << " - till " << i.till << std::endl;
    }
    proxy = session.readIov("MyTag2");
    readIov(proxy, 1, false);
    readIov(proxy, 100, true);
    session.transaction().commit();

  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
  std::cout << "## Run successfully completed." << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  int ret = 0;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::string connectionString0("sqlite_file:cms_conditions_2.db");
  ret = run(connectionString0);
  return ret;
}
