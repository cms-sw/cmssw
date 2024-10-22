#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/Common/interface/BasicPayload.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace cond::persistency;

int run(const std::string& connectionString, bool generate) {
  try {
    //*************
    std::cout << "> Connecting with db in " << connectionString << std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity(coral::Debug);
    Session session = connPool.createSession(connectionString, true);
    session.transaction().start(false);
    IOVEditor editor;
    if (!session.existsDatabase() || !session.existsIov("BasicPayload_v0")) {
      editor = session.createIov<cond::BasicPayload>("BasicPayload_v0", cond::runnumber);
      editor.setDescription("Test for timestamp selection");
    }
    for (int i = 0; i < 10; i++) {
      cond::BasicPayload p;
      if (!generate) {
        cond::BasicPayload p0(i * 10.1, i + 1., 100);
        for (size_t j = 0; j < 100; j++)
          p0.m_vec[j] = 1;
        for (size_t j = 3; j < 7; j++)
          for (size_t i = 3; i < 7; i++)
            p0.m_vec[j * 10 + i] = 0;
        p = p0;
      } else {
        cond::BasicPayload p1(i * 10.1, i + 1., 10000);
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist(0., 1.);
        for (int i = 0; i < 10000; ++i)
          p1.m_vec[i] = dist(mt);
        p = p1;
      }
      auto pid = session.storePayload(p);
      editor.insert(i * 100 + 1, pid);
      p.print();
    }
    std::cout << "flushing..." << std::endl;
    editor.flush();
    std::cout << "> iov changes flushed..." << std::endl;
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
  bool generate = false;
  if (argc > 1) {
    std::string option = std::string(argv[1]);
    generate = (option == "generate");
  }

  int ret = 0;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::string connectionString0("sqlite_file:BasicPayload_v0.db");
  ret = run(connectionString0, generate);
  return ret;
}
