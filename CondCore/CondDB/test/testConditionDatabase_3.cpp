#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
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

int run(const std::string& connectionString) {
  try {
    //*************
    std::cout << "> Connecting with db in " << connectionString << std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    Session session = connPool.createSession(connectionString, true);
    session.transaction().start(false);
    MyTestData d0(17);
    MyTestData d1(999);
    std::cout << "> Storing payload ptr=" << &d0 << std::endl;
    cond::Hash p0 = session.storePayload(d0, boost::posix_time::microsec_clock::universal_time());
    cond::Hash p1 = session.storePayload(d1, boost::posix_time::microsec_clock::universal_time());

    IOVEditor editor;
    if (!session.existsIov("MyNewIOV")) {
      editor = session.createIov<MyTestData>("MyNewIOV", cond::lumiid, cond::SYNCH_HLT);
      editor.setDescription("Test with MyTestData class");
      editor.insert(1, p0);
      editor.insert(cond::time::lumiTime(100, 11), p1);
      editor.insert(cond::time::lumiTime(100, 21), p0);
      editor.insert(cond::time::lumiTime(100, 31), p1);
      editor.insert(cond::time::lumiTime(200, 11), p1);
      editor.insert(cond::time::lumiTime(200, 21), p0);
      editor.insert(cond::time::lumiTime(200, 31), p1);
      editor.insert(cond::time::lumiTime(300, 11), p1);
      editor.insert(cond::time::lumiTime(300, 21), p0);
      editor.insert(cond::time::lumiTime(300, 31), p1);
      editor.insert(cond::time::lumiTime(400, 11), p0);
      editor.insert(cond::time::lumiTime(400, 12), p1);
      editor.insert(cond::time::lumiTime(400, 13), p0);
      std::cout << "> inserted iovs..." << std::endl;
      editor.flush();
      std::cout << "> iov changes flushed..." << std::endl;
    }

    session.transaction().commit();
    std::cout << "> iov changes committed!..." << std::endl;

    ::sleep(2);
    session.transaction().start();

    auto arr0 = session.readIov("MyNewIOV").selectAll();
    std::cout << "# Selecting all iovs..." << std::endl;
    for (const auto& iiov : arr0) {
      std::cout << "# since=" << iiov.since << " till:" << iiov.till << std::endl;
    }
    auto arr1 = session.readIov("MyNewIOV").selectRange(cond::time::lumiTime(100, 15), cond::time::lumiTime(300, 15));
    std::cout << "# Selecting range (" << cond::time::lumiTime(100, 15) << "," << cond::time::lumiTime(300, 15) << ")"
              << std::endl;
    for (const auto& iiov : arr1) {
      std::cout << "# since=" << iiov.since << " till:" << iiov.till << std::endl;
    }
    auto pxn = session.readIov("MyNewIOV");
    std::vector<cond::Time_t> inputTimes{10,
                                         cond::time::lumiTime(100, 15),
                                         cond::time::lumiTime(100, 25),
                                         cond::time::lumiTime(100, 35),
                                         cond::time::lumiTime(200, 15),
                                         cond::time::lumiTime(200, 25),
                                         cond::time::lumiTime(200, 35),
                                         cond::time::lumiTime(300, 15),
                                         cond::time::lumiTime(300, 25),
                                         cond::time::lumiTime(300, 35),
                                         cond::time::lumiTime(400, 11),
                                         cond::time::lumiTime(400, 12),
                                         cond::time::lumiTime(400, 13)};
    for (auto t : inputTimes) {
      cond::Iov_t iiov = pxn.getInterval(t);
      std::cout << "#Target=" << t << " since=" << iiov.since << " till:" << iiov.till << std::endl;
    }

    std::cout << "#Nqueries:" << pxn.numberOfQueries() << std::endl;

    session.transaction().commit();

    cond::Iov_t iov;
    auto requests = std::make_shared<std::vector<cond::Iov_t>>();
    PayloadProxy<MyTestData> ppn(&iov, &session, &requests);
    session.transaction().start(true);
    auto iovP = session.readIov("MyNewIOV");
    for (auto t : inputTimes) {
      iov = iovP.getInterval(t);
      ppn.initializeForNewIOV();
      ppn.make();
      std::cout << "PP: target=" << t << " since=" << iov.since << " till:" << iov.till << std::endl;
    }
    session.transaction().commit();

    std::cout << "#PP: nqueries:" << iovP.numberOfQueries() << std::endl;

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
  std::string connectionString0("sqlite_file:cms_conditions_3.db");
  ret = run(connectionString0);
  return ret;
}
