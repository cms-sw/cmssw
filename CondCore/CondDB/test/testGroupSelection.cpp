#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
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

int run(const std::string& csWrite, const std::string& csRead) {
  try {
    //*************
    std::cout << "> Connecting with db in " << csWrite << std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    connPool.setAuthenticationPath("/build/gg");
    Session session = connPool.createSession(csWrite, true);
    session.transaction().start(false);
    MyTestData d0(1000);
    cond::Hash p0 = session.storePayload(d0);
    MyTestData d1(1001);
    cond::Hash p1 = session.storePayload(d1);
    MyTestData d2(1002);
    cond::Hash p2 = session.storePayload(d2);
    MyTestData d3(1003);
    cond::Hash p3 = session.storePayload(d3);
    MyTestData d4(1004);
    cond::Hash p4 = session.storePayload(d4);
    MyTestData d5(1005);
    cond::Hash p5 = session.storePayload(d5);
    MyTestData d6(1006);
    cond::Hash p6 = session.storePayload(d6);
    MyTestData d7(1007);
    cond::Hash p7 = session.storePayload(d7);
    MyTestData d8(1008);
    cond::Hash p8 = session.storePayload(d8);
    IOVEditor editor;
    std::string tag("MyTestData_ts_v0");
    if (!session.existsIov(tag)) {
      editor = session.createIov<MyTestData>(tag, cond::timestamp, cond::SYNCH_ANY);
      editor.setDescription("Test for group selection");
      MyTestData dummy(0);
      cond::Hash pd = session.storePayload(dummy);
      editor.insert(1, pd);
    } else {
      editor = session.editIov(tag);
    }
    boost::posix_time::ptime tb0 = boost::posix_time::second_clock::local_time();
    boost::posix_time::ptime tb1 = tb0 + boost::posix_time::seconds(600);
    boost::posix_time::ptime tb2 = tb0 + boost::posix_time::seconds(1200);
    boost::posix_time::ptime tb3 = tb0 + boost::posix_time::seconds(1800);
    boost::posix_time::ptime tb4 = tb0 + boost::posix_time::seconds(2400);
    boost::posix_time::ptime tb5 = tb0 + boost::posix_time::seconds(3000);
    boost::posix_time::ptime tb6 = tb0 + boost::posix_time::seconds(3600);
    boost::posix_time::ptime tb7 = tb0 + boost::posix_time::seconds(4200);
    boost::posix_time::ptime tb8 = tb0 + boost::posix_time::seconds(4800);
    cond::Time_t t0 = cond::time::from_boost(tb0);
    cond::Time_t t1 = cond::time::from_boost(tb1);
    cond::Time_t t2 = cond::time::from_boost(tb2);
    cond::Time_t t3 = cond::time::from_boost(tb3);
    cond::Time_t t4 = cond::time::from_boost(tb4);
    cond::Time_t t5 = cond::time::from_boost(tb5);
    cond::Time_t t6 = cond::time::from_boost(tb6);
    cond::Time_t t7 = cond::time::from_boost(tb7);
    cond::Time_t t8 = cond::time::from_boost(tb8);
    editor.insert(t0, p0);
    editor.insert(t1, p1);
    editor.insert(t2, p2);
    editor.insert(t3, p3);
    editor.insert(t4, p4);
    editor.insert(t5, p5);
    editor.insert(t6, p6);
    editor.insert(t7, p7);
    editor.insert(t8, p8);
    editor.flush();
    session.transaction().commit();
    std::cout << "> Connecting with db in " << csRead << std::endl;
    session = connPool.createSession(csRead);
    session.transaction().start();
    IOVProxy reader = session.readIov(tag);
    cond::Time_t tg0 = cond::time::from_boost(tb0 + boost::posix_time::seconds(300));
    cond::Iov_t iov0 = reader.getInterval(tg0);
    std::cout << "tg0: since " << iov0.since << " till " << iov0.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg1 = cond::time::from_boost(tb1 + boost::posix_time::seconds(300));
    cond::Iov_t iov1 = reader.getInterval(tg1);
    std::cout << "tg1: since " << iov1.since << " till " << iov1.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg2 = cond::time::from_boost(tb2 + boost::posix_time::seconds(300));
    cond::Iov_t iov2 = reader.getInterval(tg2);
    std::cout << "tg2: since " << iov2.since << " till " << iov2.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg3 = cond::time::from_boost(tb3 + boost::posix_time::seconds(300));
    cond::Iov_t iov3 = reader.getInterval(tg3);
    std::cout << "tg3: since " << iov3.since << " till " << iov3.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg4 = cond::time::from_boost(tb4 + boost::posix_time::seconds(300));
    cond::Iov_t iov4 = reader.getInterval(tg4);
    std::cout << "tg4: since " << iov4.since << " till " << iov4.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg5 = cond::time::from_boost(tb5 + boost::posix_time::seconds(300));
    cond::Iov_t iov5 = reader.getInterval(tg5);
    std::cout << "tg5: since " << iov5.since << " till " << iov5.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg6 = cond::time::from_boost(tb6 + boost::posix_time::seconds(300));
    cond::Iov_t iov6 = reader.getInterval(tg6);
    std::cout << "tg6: since " << iov6.since << " till " << iov6.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg7 = cond::time::from_boost(tb7 + boost::posix_time::seconds(300));
    cond::Iov_t iov7 = reader.getInterval(tg7);
    std::cout << "tg7: since " << iov7.since << " till " << iov7.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    cond::Time_t tg8 = cond::time::from_boost(tb8 + boost::posix_time::seconds(300));
    cond::Iov_t iov8 = reader.getInterval(tg8);
    std::cout << "tg8: since " << iov8.since << " till " << iov8.till << " nqueries " << reader.numberOfQueries()
              << std::endl;
    session.transaction().commit();
  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }

  return 0;
}

int main(int argc, char** argv) {
  int ret = 0;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::string connectionString("sqlite_file:GroupSelection.db");
  ret = run(connectionString, connectionString);
  return ret;
}
