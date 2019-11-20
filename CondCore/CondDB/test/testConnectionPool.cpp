//Framework includes
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//Module includes
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/Types.h"
//Entity class
#include "CondFormats/RunInfo/interface/RunInfo.h"
//CORAL includes
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IColumn.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
//BOOST includes
//
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>

void testCreateCoralSession(cond::persistency::ConnectionPool& connPool,
                            std::string const& connectionString,
                            bool const writeCapable) {
  std::shared_ptr<coral::ISessionProxy> session = connPool.createCoralSession(connectionString, writeCapable);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  std::string tagTable("TAG");
  std::string iovTable("IOV");
  std::string payloadTable("PAYLOAD");
  std::string tag("RunInfo_v1_mc");
  std::string object("RunInfo");
  std::string hash("cfd8987f899e99de69626e8a91b5c6b1506b82de");
  std::unique_ptr<coral::IQuery> query(schema.tableHandle(tagTable).newQuery());
  query->addToOutputList("OBJECT_TYPE");
  query->defineOutputType("OBJECT_TYPE", "string");
  std::string tagWhereClause("NAME=:tag");
  coral::AttributeList tagBindVariableList;
  tagBindVariableList.extend("tag", typeid(std::string));
  tagBindVariableList["tag"].data<std::string>() = tag;
  query->setCondition(tagWhereClause, tagBindVariableList);
  coral::ICursor& tagCursor = query->execute();
  while (tagCursor.next()) {
    tagCursor.currentRow().toOutputStream(std::cout) << std::endl;
  }
  query.reset(schema.tableHandle(iovTable).newQuery());
  query->addToOutputList("SINCE");
  query->defineOutputType("SINCE", "unsigned long long");
  std::string iovWhereClause("TAG_NAME=:tag");
  coral::AttributeList iovBindVariableList;
  iovBindVariableList.extend("tag", typeid(std::string));
  iovBindVariableList["tag"].data<std::string>() = tag;
  query->setCondition(iovWhereClause, iovBindVariableList);
  coral::ICursor& iovCursor = query->execute();
  while (iovCursor.next()) {
    iovCursor.currentRow().toOutputStream(std::cout) << std::endl;
  }
  query.reset(schema.tableHandle(payloadTable).newQuery());
  query->addToOutputList("OBJECT_TYPE");
  query->defineOutputType("OBJECT_TYPE", "string");
  std::string payloadWhereClause("HASH=:hash");
  coral::AttributeList payloadBindVariableList;
  payloadBindVariableList.extend("hash", typeid(std::string));
  payloadBindVariableList["hash"].data<std::string>() = hash;
  query->setCondition(payloadWhereClause, payloadBindVariableList);
  coral::ICursor& payloadCursor = query->execute();
  while (payloadCursor.next()) {
    payloadCursor.currentRow().toOutputStream(std::cout) << std::endl;
  }
  session->transaction().commit();
}

void testCreateSession(cond::persistency::ConnectionPool& connPool,
                       std::string const& connectionString,
                       bool const writeCapable) {
  cond::Iov_t iov;
  cond::persistency::Session session = connPool.createSession(connectionString, writeCapable);
  auto requests = std::make_shared<std::vector<cond::Iov_t>>();
  cond::persistency::PayloadProxy<RunInfo> pp(&iov, &session, &requests);

  session.transaction().start(true);
  cond::persistency::IOVProxy iovProxy = session.readIov("RunInfo_v1_mc");
  session.transaction().commit();

  session.transaction().start(true);
  auto it = iovProxy.find(1);
  if (it != iovProxy.end()) {
    iov = *it;
  }
  session.transaction().commit();

  pp.initializeForNewIOV();
  pp.make();
  std::cout << "run number: " << pp().m_run << std::endl;
}

void testCreateReadOnlySession(cond::persistency::ConnectionPool& connPool,
                               std::string const& connectionString,
                               std::string const& transactionId) {
  cond::persistency::Session session = connPool.createReadOnlySession(connectionString, transactionId);
  session.transaction().start();
  cond::persistency::IOVProxy iov = session.readIov("RunInfo_v1_mc", true);
  std::cout << "Loaded size=" << iov.loadedSize() << std::endl;
  cond::Iov_t currentIov = *(iov.find(1));
  std::cout << "run number: " << session.fetchPayload<RunInfo>(currentIov.payloadId)->m_run << std::endl;
  session.transaction().commit();
}

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  const edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
  const edm::ServiceRegistry::Operate operate(services);

  std::array<std::string, 2> connectionStrings{
      {"frontier://FrontierPrep/CMS_CONDITIONS",
       "frontier://(proxyconfigurl=http://grid-wpad/wpad.dat)(backupproxyurl=http://"
       "cmst0frontier.cern.ch:3128)(backupproxyurl=http://cmst0frontier1.cern.ch:3128)(backupproxyurl=http://"
       "cmst0frontier2.cern.ch:3128)(backupproxyurl=http://cmsbpfrontier.cern.ch:3128)(backupproxyurl=http://"
       "cmsbpfrontier1.cern.ch:3128)(backupproxyurl=http://cmsbpfrontier2.cern.ch:3128)(backupproxyurl=http://"
       "cmsbproxy.fnal.gov:3128)(serverurl=http://cmsfrontier.cern.ch:8000/FrontierPrep)(serverurl=http://"
       "cmsfrontier1.cern.ch:8000/FrontierPrep)(serverurl=http://cmsfrontier2.cern.ch:8000/"
       "FrontierPrep)(serverurl=http://cmsfrontier3.cern.ch:8000/FrontierPrep)(serverurl=http://"
       "cmsfrontier4.cern.ch:8000/FrontierPrep)/CMS_CONDITIONS"}};
  try {
    //*************
    for (const auto& connectionString : connectionStrings) {
      std::cout << "# Connecting with db in '" << connectionString << "'" << std::endl;
      cond::persistency::ConnectionPool connPool;
      //connPool.setMessageVerbosity( coral::Debug );
      //connPool.configure();
      try {
        connPool.createCoralSession(connectionString, true);
      } catch (const std::exception& e) {
        std::cout << "EXPECTED EXCEPTION: " << e.what() << std::endl;
      }
      testCreateCoralSession(connPool, connectionString, false);
      testCreateSession(connPool, connectionString, false);
      testCreateReadOnlySession(connPool, connectionString, "");
      testCreateReadOnlySession(connPool, connectionString, "testConnectionPool");
      connPool.setFrontierSecurity("foo");
      connPool.configure();
      try {
        connPool.createCoralSession(connectionString, false);
      } catch (const cms::Exception& e) {
        std::cout << "EXPECTED EXCEPTION: " << e.what() << std::endl;
      }
      edm::ParameterSet dbParameters;
      dbParameters.addUntrackedParameter("authenticationPath", std::string(""));
      dbParameters.addUntrackedParameter("authenticationSystem", 0);
      dbParameters.addUntrackedParameter("messageLevel", 3);
      dbParameters.addUntrackedParameter("security", std::string("sig"));
      dbParameters.addUntrackedParameter("logging", false);
      connPool.setParameters(dbParameters);
      connPool.configure();
      testCreateCoralSession(connPool, connectionString, false);
      testCreateSession(connPool, connectionString, false);
      testCreateReadOnlySession(connPool, connectionString, "");
      testCreateReadOnlySession(connPool, connectionString, "testConnectionPool");
    }
  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}
