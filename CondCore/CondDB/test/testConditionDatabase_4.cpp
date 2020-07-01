#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
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

int run(const std::string& connectionString) {
  try {
    //*************
    std::cout << "> Connecting with db in " << connectionString << std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    Session session = connPool.createSession(connectionString, true);
    session.transaction().start(false);
    std::string d0("abcd1234");
    std::string d1("abcdefghil");
    cond::Hash p0 = session.storePayload(d0, boost::posix_time::microsec_clock::universal_time());
    cond::Hash p1 = session.storePayload(d0, boost::posix_time::microsec_clock::universal_time());

    IOVEditor editor;

    if (!session.existsIov("StringData")) {
      editor = session.createIov<std::string>("StringData", cond::timestamp);
      editor.setDescription("Test with std::string class");
      editor.insert(1000000, p0);
      editor.insert(2000000, p1);
      editor.flush();
    }
    session.transaction().commit();
    std::cout << "> iov changes committed!..." << std::endl;

    session.transaction().start();
    IOVProxy proxy = session.readIov("StringData");
    auto iovs = proxy.selectAll();
    session.transaction().commit();
    std::cout << "(0) Found " << iovs.size() << " iovs." << std::endl;
    for (const auto& iov : iovs) {
      std::cout << "Iov since " << iov.since << " hash " << iov.payloadId << std::endl;
    }

    session.transaction().start(false);
    cond::Hash p3 = session.storePayload(std::string("013456789"), boost::posix_time::microsec_clock::universal_time());
    editor = session.editIov("StringData");
    editor.insert(3000000, p3);
    editor.erase(2000000, p1);
    editor.flush();
    std::cout << "2nd iovs changes completed." << std::endl;
    session.transaction().commit();

    ::sleep(2);
    session.transaction().start();
    proxy = session.readIov("StringData");
    iovs = proxy.selectAll();
    session.transaction().commit();
    std::cout << "(1) Found " << iovs.size() << " iovs." << std::endl;
    for (const auto& iov : iovs) {
      std::cout << "Iov since " << iov.since << " hash " << iov.payloadId << std::endl;
    }
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
  std::string connectionString0("sqlite_file:cms_conditions_4.db");
  ret = run(connectionString0);
  return ret;
}
