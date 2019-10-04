#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
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
    Session session = connPool.createSession(connectionString, true);
    session.transaction().start(false);
    RunInfoEditor runInfoWriter = session.editRunInfo();
    runInfoWriter.insert(1000,
                         boost::posix_time::time_from_string("2017-01-01 00:00:00.000"),
                         boost::posix_time::time_from_string("2017-01-01 01:00:00.000"));
    runInfoWriter.insert(1010,
                         boost::posix_time::time_from_string("2017-01-01 01:00:10.000"),
                         boost::posix_time::time_from_string("2017-01-01 02:00:00.000"));
    runInfoWriter.insert(1020,
                         boost::posix_time::time_from_string("2017-01-01 02:00:10.000"),
                         boost::posix_time::time_from_string("2017-01-01 03:00:00.000"));
    runInfoWriter.insert(1030,
                         boost::posix_time::time_from_string("2017-01-01 03:00:10.000"),
                         boost::posix_time::time_from_string("2017-01-01 04:00:00.000"));
    runInfoWriter.flush();
    session.transaction().commit();
    session.transaction().start(false);
    std::cout << "Last inserted: " << runInfoWriter.getLastInserted() << std::endl;
    session.transaction().commit();
    session.transaction().start(false);
    runInfoWriter.insertNew(1040,
                            boost::posix_time::time_from_string("2017-01-01 04:00:10.000"),
                            boost::posix_time::time_from_string("2017-01-01 04:00:10.000"));
    runInfoWriter.flush();
    session.transaction().commit();
    session.transaction().start();
    RunInfoProxy reader = session.getRunInfo(900, 1031);
    auto it = reader.find(1015);
    if (it != reader.end()) {
      auto rdata = *it;
      std::cout << "For target=1015 Found run=" << rdata.run << " start=" << rdata.start << " end=" << rdata.end
                << std::endl;
    } else
      std::cout << " Can't find run 1015 in the selected range" << std::endl;
    it = reader.find(1030);
    if (it != reader.end()) {
      auto rdata = *it;
      std::cout << "For target=1030 Found run=" << rdata.run << " start=" << rdata.start << " end=" << rdata.end
                << std::endl;
    } else
      std::cout << " Can't find run 1030 in the selected range" << std::endl;
    it = reader.find(1035);
    if (it != reader.end()) {
      auto rdata = *it;
      std::cout << "For target=1035 Found run=" << rdata.run << " start=" << rdata.start << " end=" << rdata.end
                << std::endl;
    } else
      std::cout << " Can't find run 1035 in the selected range" << std::endl;
    session.transaction().commit();
    session.transaction().start(false);
    runInfoWriter.insertNew(1040,
                            boost::posix_time::time_from_string("2017-01-01 04:00:10.000"),
                            boost::posix_time::time_from_string("2017-01-01 05:00:00.000"));
    runInfoWriter.flush();
    session.transaction().commit();
    session.transaction().start();
    reader = session.getRunInfo(1036, 1036);
    auto run = reader.get(1036);
    std::cout << "For target=1036 Found run=" << run.run << " start=" << run.start << " end=" << run.end << std::endl;
    run = reader.get(1037);
    std::cout << "For target=1037 Found run=" << run.run << " start=" << run.start << " end=" << run.end << std::endl;
    try {
      run = reader.get(1041);
      std::cout << "For target=1041 Found run=" << run.run << " start=" << run.start << " end=" << run.end << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Expected error:" << e.what() << std::endl;
    }
    session.transaction().commit();
    session.transaction().start();
    std::cout << "Last inserted: " << runInfoWriter.getLastInserted() << std::endl;
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
  std::string connectionString0("sqlite_file:run_info_0.db");
  ret = run(connectionString0);
  return ret;
}
