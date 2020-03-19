#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/PayloadProxy.h"
//
#include "MyTestData.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

cond::Iov_t getIovFromTag(Session& session, const std::string& tagName, cond::Time_t targetTime) {
  cond::Iov_t ret;
  session.transaction().start(true);
  auto iovP = session.readIov(tagName);
  ret = iovP.getInterval(targetTime);
  session.transaction().commit();
  return ret;
}

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string connectionString("sqlite_file:PayloadProxy.db");
  std::cout << "# Connecting with db in " << connectionString << std::endl;
  try {
    //*************
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    Session session = connPool.createSession(connectionString, true);
    session.transaction().start(false);
    MyTestData d0(20000);
    MyTestData d1(30000);
    std::cout << "# Storing payloads..." << std::endl;
    cond::Hash p0 = session.storePayload(d0, boost::posix_time::microsec_clock::universal_time());
    cond::Hash p1 = session.storePayload(d1, boost::posix_time::microsec_clock::universal_time());
    std::string d2("abcd1234");
    cond::Hash p2 = session.storePayload(d2, boost::posix_time::microsec_clock::universal_time());
    std::string d3("abcd1234");
    cond::Hash p3 = session.storePayload(d3, boost::posix_time::microsec_clock::universal_time());
    IOVEditor editor;
    if (!session.existsIov("MyNewIOV2")) {
      editor = session.createIov<MyTestData>("MyNewIOV2", cond::runnumber);
      editor.setDescription("Test with MyTestData class");
      editor.insert(1, p0);
      editor.insert(100, p1);
      std::cout << "# inserted 2 iovs..." << std::endl;
      editor.flush();
      std::cout << "# iov changes flushed..." << std::endl;
    }
    if (!session.existsIov("StringData2")) {
      editor = session.createIov<std::string>("StringData2", cond::timestamp);
      editor.setDescription("Test with std::string class");
      editor.insert(1000000, p2);
      editor.insert(2000000, p3);
      editor.flush();
    }
    if (!session.existsIov("StringData3")) {
      editor = session.createIov<std::string>("StringData3", cond::lumiid);
      editor.setDescription("Test with std::string class");
      editor.insert(4294967297, p2);
      editor.flush();
    }

    session.transaction().commit();
    std::cout << "# iov changes committed!..." << std::endl;
    ::sleep(2);

    cond::Iov_t iov0;
    auto requests0 = std::make_shared<std::vector<cond::Iov_t>>();
    PayloadProxy<MyTestData> pp0(&iov0, &session, &requests0);

    cond::Iov_t iov1;
    auto requests1 = std::make_shared<std::vector<cond::Iov_t>>();
    PayloadProxy<std::string> pp1(&iov1, &session, &requests1);

    iov0 = getIovFromTag(session, "MyNewIOV2", 25);
    pp0.initializeForNewIOV();
    pp0.make();

    const MyTestData& rd0 = pp0();
    if (rd0 != d0) {
      std::cout << "ERROR: MyTestData object read different from source." << std::endl;
    } else {
      std::cout << "MyTestData instance valid from " << iov0.since << " to " << iov0.till << std::endl;
    }

    iov0 = getIovFromTag(session, "MyNewIOV2", 35);
    pp0.initializeForNewIOV();
    pp0.make();

    const MyTestData& rd1 = pp0();
    if (rd1 != d0) {
      std::cout << "ERROR: MyTestData object read different from source." << std::endl;
    } else {
      std::cout << "MyTestData instance valid from " << iov0.since << " to " << iov0.till << std::endl;
    }

    iov0 = getIovFromTag(session, "MyNewIOV2", 100000);
    pp0.initializeForNewIOV();
    pp0.make();

    const MyTestData& rd2 = pp0();
    if (rd2 != d1) {
      std::cout << "ERROR: MyTestData object read different from source." << std::endl;
    } else {
      std::cout << "MyTestData instance valid from " << iov0.since << " to " << iov0.till << std::endl;
    }

    try {
      iov1 = getIovFromTag(session, "StringData2", 345);
    } catch (cond::persistency::Exception& e) {
      std::cout << "Expected error: " << e.what() << std::endl;
    }

    iov1 = getIovFromTag(session, "StringData2", 1000000);
    pp1.initializeForNewIOV();
    pp1.make();
    const std::string& rd3 = pp1();
    if (rd3 != d2) {
      std::cout << "ERROR: std::string object read different from source." << std::endl;
    } else {
      std::cout << "std::string instance valid from " << iov1.since << " to " << iov1.till << std::endl;
    }

    iov1 = getIovFromTag(session, "StringData2", 3000000);
    pp1.initializeForNewIOV();
    pp1.make();
    const std::string& rd4 = pp1();
    if (rd4 != d3) {
      std::cout << "ERROR: std::string object read different from source." << std::endl;
    } else {
      std::cout << "std::string instance valid from " << iov1.since << " to " << iov1.till << std::endl;
    }

    cond::Iov_t iov2;
    auto requests2 = std::make_shared<std::vector<cond::Iov_t>>();
    PayloadProxy<std::string> pp2(&iov2, &session, &requests2);

    try {
      iov2 = getIovFromTag(session, "StringData3", 3000000);
    } catch (cond::persistency::Exception& e) {
      std::cout << "Expected error: " << e.what() << std::endl;
    }
  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}
