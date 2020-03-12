#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
//
#include "CondCore/CondDB/interface/CondDB.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::db;

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string connectionString0("oracle://cms_orcon_adg/CMS_COND_31X_RUN_INFO");
  std::string connectionString1("oracle://cms_orcoff_prep/CMS_CONDITIONS");
  std::string gtConnectionStr("oracle://cms_orcon_adg/CMS_COND_31X_GLOBALTAG");
  try {
    Session session;
    //session.configuration().setMessageVerbosity( coral::Debug );
    size_t i = 0;
    IOVProxy reader;
    std::cout << "# Connecting with db in " << connectionString0 << std::endl;
    session.open(connectionString0, true);
    session.transaction().start(true);
    std::string tag0("runinfo_31X_hlt");
    reader = session.readIov(tag0, true);
    std::cout << "Tag " << reader.tag() << " timeType:" << cond::time::timeTypeName(reader.timeType())
              << " size:" << reader.size() << " type:" << reader.payloadObjectType()
              << " endOfValidity:" << reader.endOfValidity() << " lastValidatedTime:" << reader.lastValidatedTime()
              << std::endl;

    for (auto iov : reader) {
      i++;
      std::cout << "#Since " << iov.since << " Till " << iov.till << " PID " << iov.payloadId << std::endl;
      if (i == 20)
        break;
    }
    session.transaction().commit();
    session.close();

    std::cout << "# Connecting with db in " << connectionString1 << std::endl;
    session.open(connectionString1, true);
    session.transaction().start(true);
    reader = session.readIov(tag0, true);
    i = 0;
    std::cout << "Tag " << reader.tag() << " timeType:" << cond::time::timeTypeName(reader.timeType())
              << " size:" << reader.size() << " type:" << reader.payloadObjectType()
              << " endOfValidity:" << reader.endOfValidity() << " lastValidatedTime:" << reader.lastValidatedTime()
              << std::endl;

    for (auto iov : reader) {
      i++;
      std::cout << "#Since " << iov.since << " Till " << iov.till << " PID " << iov.payloadId << std::endl;
      if (i == 20)
        break;
    }
    session.transaction().commit();
    session.close();

    std::cout << "# Connecting with db in " << gtConnectionStr << std::endl;
    session.open(gtConnectionStr, true);
    session.transaction().start(true);
    GTProxy gtReader = session.readGlobalTag("FT_R_53_V6");
    i = 0;
    for (auto t : gtReader) {
      i++;
      std::cout << "#Tag " << t.tagName() << " Record " << t.recordName() << " Label " << t.recordLabel() << std::endl;
      if (i == 20)
        break;
    }
    session.transaction().commit();
    session.close();

    std::cout << "# Connecting with db in " << connectionString1 << std::endl;
    session.open(connectionString1, true);
    session.transaction().start(true);
    gtReader = session.readGlobalTag("FT_R_53_V6");
    i = 0;
    for (auto t : gtReader) {
      i++;
      std::cout << "#Tag " << t.tagName() << " Record " << t.recordName() << " Label " << t.recordLabel() << std::endl;
      if (i == 20)
        break;
    }
    session.transaction().commit();
    session.close();

  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}
