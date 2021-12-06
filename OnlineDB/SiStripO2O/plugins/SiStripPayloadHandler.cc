#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <openssl/sha.h>

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/TimeStamp.h"

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"

template <typename SiStripPayload>
class SiStripPayloadHandler : public edm::EDAnalyzer {
public:
  explicit SiStripPayloadHandler(const edm::ParameterSet& iConfig);
  ~SiStripPayloadHandler() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;

private:
  std::string makeConfigHash();
  std::string queryConfigMap(std::string configHash);
  void updateConfigMap(std::string configHash, std::string payloadHash);

private:
  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_configMapDb;
  std::string m_cfgMapSchemaName;
  std::string m_cfgMapTableName;
  std::string m_condDb;
  std::string m_cfgMapDbFile;
  std::string m_localCondDbFile;
  std::string m_targetTag;
  cond::Time_t m_since;

  std::string p_type;
  std::string p_cfgstr;
};

template <typename SiStripPayload>
SiStripPayloadHandler<SiStripPayload>::SiStripPayloadHandler(const edm::ParameterSet& iConfig)
    : m_connectionPool(),
      m_cfgMapSchemaName(iConfig.getUntrackedParameter<std::string>("cfgMapSchemaName", "CMS_COND_O2O")),
      m_cfgMapTableName(iConfig.getUntrackedParameter<std::string>("cfgMapTableName", "STRIP_CONFIG_TO_PAYLOAD_MAP")),
      m_condDb(iConfig.getParameter<std::string>("conditionDatabase")),
      m_localCondDbFile(iConfig.getParameter<std::string>("condDbFile")),
      m_targetTag(iConfig.getParameter<std::string>("targetTag")),
      m_since(iConfig.getParameter<uint32_t>("since")),
      p_type(cond::demangledName(typeid(SiStripPayload))) {
  edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;
  p_cfgstr = condObjBuilder->getConfigString(typeid(SiStripPayload));
  if (iConfig.exists("configMapDatabase"))
    m_configMapDb = iConfig.getParameter<std::string>("configMapDatabase");
  if (iConfig.exists("cfgMapDbFile"))
    m_cfgMapDbFile = iConfig.getParameter<std::string>("cfgMapDbFile");
  m_connectionPool.setParameters(iConfig.getParameter<edm::ParameterSet>("DBParameters"));
  m_connectionPool.configure();
}

template <typename SiStripPayload>
SiStripPayloadHandler<SiStripPayload>::~SiStripPayloadHandler() {}

template <typename SiStripPayload>
void SiStripPayloadHandler<SiStripPayload>::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  // some extra work: Getting the payload hash of the last IOV from condDb
  // Will be compared with the new payload and reported
  cond::persistency::Session condDbSession = m_connectionPool.createSession(m_condDb);
  condDbSession.transaction().start(true);
  cond::persistency::IOVProxy iovProxy = condDbSession.readIov(m_targetTag);
  cond::Hash last_hash = iovProxy.getLast().payloadId;

  // that's the final goal: obtain the payload to store into the sqlite file for the upload into the DB
  std::shared_ptr<SiStripPayload> payloadToUpload;
  // first compute the hash of the configuration
  std::string configHash = makeConfigHash();
  // query the configMap DB to find the corresponding payload hash
  std::string mappedPayloadHash = queryConfigMap(configHash);
  bool mapUpToDate = false;

  if (!mappedPayloadHash.empty()) {
    // the payload has been found ( fast O2O case )
    // copy the payload from condtition database
    mapUpToDate = true;
    edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                          << "Try retrieving payload from " << m_condDb;
    payloadToUpload = condDbSession.fetchPayload<SiStripPayload>(mappedPayloadHash);
    std::cout << "@@@[FastO2O:true]@@@" << std::endl;
    edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                          << " ... Payload is copied from offline condition database.";
  } else {
    // start the long O2O...
    edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                          << "NO mapping payload hash found. Will run the long O2O. ";
    SiStripPayload* obj = nullptr;
    edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;
    if (typeid(SiStripPayload) == typeid(SiStripApvGain)) {
      // special treatment for ApvGain : provide last payload in DB
      condObjBuilder->setLastIovGain(condDbSession.fetchPayload<SiStripApvGain>(last_hash));
    }
    condObjBuilder->getValue(obj);
    payloadToUpload = std::shared_ptr<SiStripPayload>(obj);
    std::cout << "@@@[FastO2O:false]@@@" << std::endl;
    edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                          << " ... New payload has been created.";
  }
  condDbSession.transaction().commit();

  // write payload and iov in the local file
  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                        << "Write payload to local sqlite file: " << m_localCondDbFile;
  cond::persistency::Session localFileSession = m_connectionPool.createSession(m_localCondDbFile, true);
  localFileSession.transaction().start(false);
  // write the payload
  cond::Hash thePayloadHash = localFileSession.storePayload<SiStripPayload>(*payloadToUpload);
  cond::persistency::IOVEditor iovEditor = localFileSession.createIov<SiStripPayload>(m_targetTag, cond::runnumber);
  iovEditor.setDescription("New IOV");
  // inserting the iov
  iovEditor.insert(m_since, thePayloadHash);
  iovEditor.flush();
  localFileSession.transaction().commit();
  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                        << "Payload " << thePayloadHash << " inserted to sqlite with IOV " << m_since;

  // last step, update the configMap if required
  if (!mapUpToDate) {
    updateConfigMap(configHash, thePayloadHash);
  }

  // finish the extra work: Compare the new payload with last IOV
  if (last_hash == thePayloadHash) {
    std::cout << "@@@[PayloadChange:false]@@@" << last_hash << std::endl;
  } else {
    std::cout << "@@@[PayloadChange:true]@@@" << last_hash << " -> " << thePayloadHash << std::endl;
  }
}

template <typename SiStripPayload>
void SiStripPayloadHandler<SiStripPayload>::endJob() {}

template <typename SiStripPayload>
std::string SiStripPayloadHandler<SiStripPayload>::makeConfigHash() {
  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                        << "Convert config string to SHA-1 hash for " << p_type
                                        << "\n... config: " << p_cfgstr;

  // calcuate SHA-1 hash using openssl
  // adapted from cond::persistency::makeHash() in CondCore/CondDB/src/IOVSchema.cc
  SHA_CTX ctx;
  if (!SHA1_Init(&ctx)) {
    throw cms::Exception("SHA1 initialization error.");
  }
  if (!SHA1_Update(&ctx, p_type.c_str(), p_type.size())) {
    throw cms::Exception("SHA1 processing error (1).");
  }
  if (!SHA1_Update(&ctx, p_cfgstr.c_str(), p_cfgstr.size())) {
    throw cms::Exception("SHA1 processing error (2).");
  }
  unsigned char hash[SHA_DIGEST_LENGTH];
  if (!SHA1_Final(hash, &ctx)) {
    throw cms::Exception("SHA1 finalization error.");
  }

  char tmp[SHA_DIGEST_LENGTH * 2 + 1];
  // re-write bytes in hex
  for (unsigned int i = 0; i < SHA_DIGEST_LENGTH; i++) {
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);
  }
  tmp[SHA_DIGEST_LENGTH * 2] = 0;

  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__
                                        << "] "
                                           "... hash: "
                                        << tmp;
  return tmp;
}

template <typename SiStripPayload>
std::string SiStripPayloadHandler<SiStripPayload>::queryConfigMap(std::string configHash) {
  if (m_configMapDb.empty())
    return "";  // return empty string if m_configMapDb is not specified

  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                        << "Query " << m_configMapDb << " to see if the payload is already in DB.";

  auto cmDbSession = m_connectionPool.createCoralSession(m_configMapDb);
  // query the STRIP_CONFIG_TO_PAYLOAD_MAP table
  cmDbSession->transaction().start(true);
  coral::ITable& cmTable = cmDbSession->schema(m_cfgMapSchemaName).tableHandle(m_cfgMapTableName);
  std::unique_ptr<coral::IQuery> query(cmTable.newQuery());
  query->addToOutputList("PAYLOAD_HASH");
  query->defineOutputType("PAYLOAD_HASH", coral::AttributeSpecification::typeNameForType<std::string>());
  // also print these for debugging
  query->addToOutputList("PAYLOAD_TYPE");
  query->addToOutputList("CONFIG_STRING");
  query->addToOutputList("INSERTION_TIME");
  std::string whereClause("CONFIG_HASH = :CONFIG_HASH");
  coral::AttributeList whereData;
  whereData.extend<std::string>("CONFIG_HASH");
  whereData.begin()->data<std::string>() = configHash;
  query->setCondition(whereClause, whereData);
  coral::ICursor& cursor = query->execute();
  std::string p_hash;
  if (cursor.next()) {
    // the payload has been found ( fast O2O case )
    p_hash = cursor.currentRow()["PAYLOAD_HASH"].data<std::string>();
    edm::LogInfo("SiStripPayloadHandler")
        << "[SiStripPayloadHandler::" << __func__ << "] "
        << "Found associated payload hash " << p_hash
        << "\n...  type=" << cursor.currentRow()["PAYLOAD_TYPE"].data<std::string>()
        << "\n...  config=" << cursor.currentRow()["CONFIG_STRING"].data<std::string>()
        << "\n...  insertion_time=" << cursor.currentRow()["INSERTION_TIME"].data<coral::TimeStamp>().toString();
  }
  cmDbSession->transaction().commit();

  return p_hash;
}

template <typename SiStripPayload>
void SiStripPayloadHandler<SiStripPayload>::updateConfigMap(std::string configHash, std::string payloadHash) {
  if (m_cfgMapDbFile.empty())
    return;  // skip this if m_cfgMapDbFile is not set

  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                        << "Updating the config to payload hash map to " << m_cfgMapDbFile;

  // create a writable transaction
  auto cmSQLiteSession = m_connectionPool.createCoralSession(m_cfgMapDbFile, true);
  cmSQLiteSession->transaction().start(false);

  if (!cmSQLiteSession->nominalSchema().existsTable(m_cfgMapTableName)) {
    // create the table if it does not exist
    coral::TableDescription mapTable;
    mapTable.setName(m_cfgMapTableName);
    mapTable.insertColumn("CONFIG_HASH", coral::AttributeSpecification::typeNameForType<std::string>());
    mapTable.insertColumn("PAYLOAD_HASH", coral::AttributeSpecification::typeNameForType<std::string>());
    mapTable.insertColumn("PAYLOAD_TYPE", coral::AttributeSpecification::typeNameForType<std::string>());
    mapTable.insertColumn("CONFIG_STRING", coral::AttributeSpecification::typeNameForType<std::string>());
    mapTable.insertColumn("INSERTION_TIME", coral::AttributeSpecification::typeNameForType<coral::TimeStamp>());
    mapTable.setPrimaryKey("CONFIG_HASH");
    mapTable.setNotNullConstraint("CONFIG_HASH");
    mapTable.setNotNullConstraint("PAYLOAD_HASH");
    mapTable.setNotNullConstraint("PAYLOAD_TYPE");
    mapTable.setNotNullConstraint("CONFIG_STRING");
    mapTable.setNotNullConstraint("INSERTION_TIME");
    cmSQLiteSession->nominalSchema().createTable(mapTable);
  }

  coral::ITable& cmTable = cmSQLiteSession->nominalSchema().tableHandle(m_cfgMapTableName);
  coral::AttributeList insertData;
  insertData.extend<std::string>("CONFIG_HASH");
  insertData.extend<std::string>("PAYLOAD_HASH");
  // also insert these for bookkeeping
  insertData.extend<std::string>("PAYLOAD_TYPE");
  insertData.extend<std::string>("CONFIG_STRING");
  insertData.extend<coral::TimeStamp>("INSERTION_TIME");
  insertData["CONFIG_HASH"].data<std::string>() = configHash;
  insertData["PAYLOAD_HASH"].data<std::string>() = payloadHash;
  insertData["PAYLOAD_TYPE"].data<std::string>() = p_type;
  insertData["CONFIG_STRING"].data<std::string>() = p_cfgstr;
  insertData["INSERTION_TIME"].data<coral::TimeStamp>() = coral::TimeStamp::now();  // UTC time
  cmTable.dataEditor().insertRow(insertData);
  cmSQLiteSession->transaction().commit();
  edm::LogInfo("SiStripPayloadHandler") << "[SiStripPayloadHandler::" << __func__ << "] "
                                        << "Updated with mapping (configHash : payloadHash)" << configHash << " : "
                                        << payloadHash;
}

// -------------------------------------------------------

typedef SiStripPayloadHandler<SiStripApvGain> SiStripO2OApvGain;
DEFINE_FWK_MODULE(SiStripO2OApvGain);

typedef SiStripPayloadHandler<SiStripBadStrip> SiStripO2OBadStrip;
DEFINE_FWK_MODULE(SiStripO2OBadStrip);

typedef SiStripPayloadHandler<SiStripFedCabling> SiStripO2OFedCabling;
DEFINE_FWK_MODULE(SiStripO2OFedCabling);

typedef SiStripPayloadHandler<SiStripLatency> SiStripO2OLatency;
DEFINE_FWK_MODULE(SiStripO2OLatency);

typedef SiStripPayloadHandler<SiStripNoises> SiStripO2ONoises;
DEFINE_FWK_MODULE(SiStripO2ONoises);

typedef SiStripPayloadHandler<SiStripPedestals> SiStripO2OPedestals;
DEFINE_FWK_MODULE(SiStripO2OPedestals);

typedef SiStripPayloadHandler<SiStripThreshold> SiStripO2OThreshold;
DEFINE_FWK_MODULE(SiStripO2OThreshold);
