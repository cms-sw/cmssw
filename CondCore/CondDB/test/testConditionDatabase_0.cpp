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
    MyTestData d0(17);
    MyTestData d1(999);
    std::cout << "> Storing payload ptr=" << &d0 << std::endl;
    cond::Hash p0 = session.storePayload(d0, boost::posix_time::microsec_clock::universal_time());
    cond::Hash p1 = session.storePayload(d1, boost::posix_time::microsec_clock::universal_time());
    std::string d("abcd1234");
    cond::Hash p3 = session.storePayload(d, boost::posix_time::microsec_clock::universal_time());

    IOVEditor editor;
    if (!session.existsIov("MyNewIOV")) {
      editor = session.createIov<MyTestData>("MyNewIOV", cond::runnumber, cond::SYNCH_OFFLINE);
      editor.setDescription("Test with MyTestData class");
      editor.insert(1, p0);
      editor.insert(100, p1);
      std::cout << "> inserted 2 iovs..." << std::endl;
      editor.flush();
      std::cout << "> iov changes flushed..." << std::endl;
    }

    if (!session.existsIov("StringData")) {
      editor = session.createIov<std::string>("StringData", cond::timestamp);
      editor.setDescription("Test with std::string class");
      editor.insert(1000000, p3);
      editor.insert(2000000, p3);
      editor.flush();
    }

    session.transaction().commit();
    std::cout << "> iov changes committed!..." << std::endl;

    session.transaction().start(false);
    std::cout << "## now trying to insert in the past..." << std::endl;
    try {
      editor = session.editIov("MyNewIOV");
      editor.insert(200, p1);
      editor.insert(300, p1);
      editor.insert(50, p1);
      editor.flush();
      std::cout << "ERROR: forbidden insertion." << std::endl;
      session.transaction().commit();
    } catch (const cond::persistency::Exception& e) {
      std::cout << "Expected error: " << e.what() << std::endl;
      session.transaction().rollback();
    }
    session.transaction().start(false);
    editor = session.editIov("StringData");
    editor.insert(3000000, p3);
    editor.insert(4000000, p3);
    editor.insert(1500000, p3);
    editor.flush();
    std::cout << "Insertion in the past completed." << std::endl;
    session.transaction().commit();

    ::sleep(2);
    session.transaction().start();

    IOVProxy proxy = session.readIov("MyNewIOV");
    std::cout << "> iov loaded size=" << proxy.loadedSize() << std::endl;
    std::cout << "> iov sequence size=" << proxy.sequenceSize() << std::endl;
    IOVArray iovs = proxy.selectAll();
    IOVArray::Iterator iovIt = iovs.find(57);
    if (iovIt == iovs.end()) {
      std::cout << ">[0] not found!" << std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout << "#[0] iov since=" << val.since << " till=" << val.till << " pid=" << val.payloadId << std::endl;
      std::shared_ptr<MyTestData> pay0 = session.fetchPayload<MyTestData>(val.payloadId);
      pay0->print();
      iovIt++;
    }
    if (iovIt == iovs.end()) {
      std::cout << "#[1] not found!" << std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout << "#[1] iov since=" << val.since << " till=" << val.till << " pid=" << val.payloadId << std::endl;
      std::shared_ptr<MyTestData> pay1 = session.fetchPayload<MyTestData>(val.payloadId);
      pay1->print();
    }
    iovIt = iovs.find(176);
    if (iovIt == iovs.end()) {
      std::cout << "#[2] not found!" << std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout << "#[2] iov since=" << val.since << " till=" << val.till << " pid=" << val.payloadId << std::endl;
      std::shared_ptr<MyTestData> pay2 = session.fetchPayload<MyTestData>(val.payloadId);
      pay2->print();
      iovIt++;
    }
    if (iovIt == iovs.end()) {
      std::cout << "#[3] not found!" << std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout << "#[3] iov since=" << val.since << " till=" << val.till << " pid=" << val.payloadId << std::endl;
      std::shared_ptr<MyTestData> pay3 = session.fetchPayload<MyTestData>(val.payloadId);
      pay3->print();
    }

    proxy = session.readIov("StringData");
    iovs = proxy.selectAll();
    auto iov2It = iovs.find(1000022);
    if (iov2It == iovs.end()) {
      std::cout << "#[4] not found!" << std::endl;
    } else {
      cond::Iov_t val = *iov2It;
      std::cout << "#[4] iov since=" << val.since << " till=" << val.till << " pid=" << val.payloadId << std::endl;
      std::shared_ptr<std::string> pay4 = session.fetchPayload<std::string>(val.payloadId);
      std::cout << "#pay4=" << *pay4 << std::endl;
    }
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
  std::string connectionString0("sqlite_file:cms_conditions_0.db");
  ret = run(connectionString0);
  return ret;
}
