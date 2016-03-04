#include "FWCore/Framework/interface/Event.h"
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
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/TimeStamp.h"

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"

template <typename SiStripPayload>
class SiStripPayloadAnalyzer : public edm::EDAnalyzer {
public:
  explicit SiStripPayloadAnalyzer(const edm::ParameterSet& iConfig );
  virtual ~SiStripPayloadAnalyzer();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  std::string makeConfigHash();

private:
  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_configMapDb;
  std::string m_condDb;
  std::string m_localCondDbFile;
  std::string m_targetTag;
  cond::Time_t m_since;

  std::string p_type;
  std::string p_cfgstr;
  edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;
};


template<typename SiStripPayload>
SiStripPayloadAnalyzer<SiStripPayload>::SiStripPayloadAnalyzer(const edm::ParameterSet& iConfig):
  m_connectionPool(),
  m_configMapDb( iConfig.getParameter< std::string >("configMapDatabase") ),
  m_condDb( iConfig.getParameter< std::string >("conditionDatabase") ),
  m_localCondDbFile( iConfig.getParameter< std::string >("condDbFile") ),
  m_targetTag( iConfig.getParameter< std::string >("targetTag") ),
  m_since( iConfig.getParameter< uint32_t >("since") ),
  p_type( cond::demangledName(typeid(SiStripPayload)) ),
  p_cfgstr( condObjBuilder->getConfigString(typeid(SiStripPayload)) ){
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();
}

template<typename SiStripPayload>
SiStripPayloadAnalyzer<SiStripPayload>::~SiStripPayloadAnalyzer() {
}

template<typename SiStripPayload>
void SiStripPayloadAnalyzer<SiStripPayload>::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup ) {

  // some extra work: Getting the payload hash of the last IOV from condDb
  // Will be compared with the new payload and reported
  cond::persistency::Session condDbSession = m_connectionPool.createSession( m_condDb );
  condDbSession.transaction().start( true );
  cond::persistency::IOVProxy iovProxy = condDbSession.readIov( m_targetTag );
  cond::Hash last_hash = iovProxy.getLast().payloadId;

  // that's the final goal: obtain the payload to store into the sqlite file for the upload into the DB
  boost::shared_ptr<SiStripPayload> payloadToUpload;
  // first compute the hash of the configuration
  std::string configHash = makeConfigHash();

  // open a session on the configMapDb
  edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Query " << m_configMapDb << " to see if the payload is already in DB.";
  boost::shared_ptr<coral::ISessionProxy> cmDbSession = m_connectionPool.createCoralSession( m_configMapDb, true );
  // query the CONFIG_TO_PAYLOAD_MAP table
  cmDbSession->transaction().start( true );
  coral::ITable& cmTable = cmDbSession->nominalSchema().tableHandle( "CONFIG_TO_PAYLOAD_MAP" );
  std::unique_ptr<coral::IQuery> query( cmTable.newQuery() );
  query->addToOutputList( "PAYLOAD_HASH" );
  query->defineOutputType( "PAYLOAD_HASH", coral::AttributeSpecification::typeNameForType<std::string>() );
  // also print these for debugging
  query->addToOutputList( "PAYLOAD_TYPE" );
  query->addToOutputList( "CONFIG_STRING" );
  query->addToOutputList( "INSERTION_TIME" );
  std::string whereClause( "CONFIG_HASH = :CONFIG_HASH" );
  coral::AttributeList whereData;
  whereData.extend<std::string>( "CONFIG_HASH" );
  whereData.begin()->data< std::string >() =  configHash;
  query->setCondition( whereClause, whereData );
  coral::ICursor& cursor = query->execute();
  bool mapUpToDate = false;
  if( cursor.next() ){
    // the payload has been found ( fast O2O case )
    cond::Hash p_hash = cursor.currentRow()["PAYLOAD_HASH"].data<std::string>();
    edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Found associated payload hash " << p_hash
        << "\n...  type=" << cursor.currentRow()["PAYLOAD_TYPE"].data<std::string>()
        << "\n...  config="  << cursor.currentRow()["CONFIG_STRING"].data<std::string>()
        << "\n...  insertion_time=" << cursor.currentRow()["INSERTION_TIME"].data<coral::TimeStamp>().toString();
    // copy the payload from condtition database
    edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Try retrieving payload from " << m_condDb;
    payloadToUpload = condDbSession.fetchPayload<SiStripPayload>( p_hash );
    mapUpToDate = true;
    std::cout << "@@@[FastO2O:true]@@@" << std::endl;
    edm::LogInfo("...") << " Payload is copied from offline condition database.";
  } else {
    // start the long O2O...
    edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "NO mapping payload hash found.";
    SiStripPayload *obj = nullptr;
    condObjBuilder->getValue(obj);
    payloadToUpload = boost::shared_ptr<SiStripPayload>(obj);
    std::cout << "@@@[FastO2O:false]@@@" << std::endl;
    edm::LogInfo("...") << " New payload has been created.";
  }
  cmDbSession->transaction().commit();
  condDbSession.transaction().commit();

  // write payload and iov in the local file
  edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Write payload to local sqlite file: " << m_localCondDbFile;
  cond::persistency::Session localFileSession = m_connectionPool.createSession( m_localCondDbFile, true );
  localFileSession.transaction().start( false );
  // write the payload
  cond::Hash thePayloadHash = localFileSession.storePayload<SiStripPayload>( *payloadToUpload );
  cond::persistency::IOVEditor iovEditor = localFileSession.createIov<SiStripPayload>( m_targetTag, cond::runnumber );
  iovEditor.setDescription( "New IOV" );
  // inserting the iov
  iovEditor.insert( m_since, thePayloadHash );
  iovEditor.flush();
  localFileSession.transaction().commit();
  edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Payload " << thePayloadHash << " inserted to sqlite with IOV " << m_since;

  // last step, update the configMap if required
  if( !mapUpToDate ){
    edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Updating the config to payload hash map...";
    // create a writable transaction
    cmDbSession->transaction().start( false );
    coral::AttributeList insertData;
    insertData.extend<std::string>( "CONFIG_HASH" );
    insertData.extend<std::string>( "PAYLOAD_HASH" );
    // also insert these for bookkeeping
    insertData.extend<std::string>( "PAYLOAD_TYPE" );
    insertData.extend<std::string>( "CONFIG_STRING" );
    insertData.extend<coral::TimeStamp>( "INSERTION_TIME" );
    insertData["CONFIG_HASH"].data<std::string>() = configHash;
    insertData["PAYLOAD_HASH"].data<std::string>() = thePayloadHash;
    insertData["PAYLOAD_TYPE"].data<std::string>() = p_type;
    insertData["CONFIG_STRING"].data<std::string>() = p_cfgstr;
    insertData["INSERTION_TIME"].data<coral::TimeStamp>() = coral::TimeStamp::now(); // UTC time
    cmTable.dataEditor().insertRow( insertData );
    cmDbSession->transaction().commit();
    edm::LogInfo("SiStripPayloadAnalyzer::analyze") << "Updated with mapping " << configHash << "-->" << thePayloadHash;
  }

  // finish the extra work: Compare the new payload with last IOV
  if (last_hash == thePayloadHash){
    std::cout << "@@@[PayloadChange:false]@@@" << last_hash << std::endl;
  }else {
    std::cout << "@@@[PayloadChange:true]@@@"  << last_hash << " -> " << thePayloadHash << std::endl;
  }

}

template<typename SiStripPayload>
void SiStripPayloadAnalyzer<SiStripPayload>::endJob() {
}

template<typename SiStripPayload>
std::string SiStripPayloadAnalyzer<SiStripPayload>::makeConfigHash() {

  edm::LogInfo("SiStripPayloadAnalyzer::makeConfigHash") << "Convert config string to SHA-1 hash for " << p_type
      << "\n... config: " << p_cfgstr;

  // calcuate SHA-1 hash using openssl
  // adapted from cond::persistency::makeHash() in CondCore/CondDB/src/IOVSchema.cc
  SHA_CTX ctx;
  if( !SHA1_Init( &ctx ) ){
    throw cms::Exception("SHA1 initialization error.");
  }
  if( !SHA1_Update( &ctx, p_type.c_str(), p_type.size() ) ){
    throw cms::Exception("SHA1 processing error (1).");
  }
  if( !SHA1_Update( &ctx, p_cfgstr.c_str(), p_cfgstr.size() ) ){
    throw cms::Exception("SHA1 processing error (2).");
  }
  unsigned char hash[SHA_DIGEST_LENGTH];
  if( !SHA1_Final(hash, &ctx) ){
    throw cms::Exception("SHA1 finalization error.");
  }

  char tmp[SHA_DIGEST_LENGTH*2+1];
  // re-write bytes in hex
  for (unsigned int i = 0; i < SHA_DIGEST_LENGTH; i++) {
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);
  }
  tmp[SHA_DIGEST_LENGTH*2] = 0;

  edm::LogInfo("... hash: ") << tmp;
  return tmp;

}

// -------------------------------------------------------

typedef SiStripPayloadAnalyzer<SiStripApvGain> SiStripO2OApvGain;
DEFINE_FWK_MODULE(SiStripO2OApvGain);

typedef SiStripPayloadAnalyzer<SiStripBadStrip> SiStripO2OBadStrip;
DEFINE_FWK_MODULE(SiStripO2OBadStrip);

typedef SiStripPayloadAnalyzer<SiStripFedCabling> SiStripO2OFedCabling;
DEFINE_FWK_MODULE(SiStripO2OFedCabling);

typedef SiStripPayloadAnalyzer<SiStripLatency> SiStripO2OLatency;
DEFINE_FWK_MODULE(SiStripO2OLatency);

typedef SiStripPayloadAnalyzer<SiStripNoises> SiStripO2ONoises;
DEFINE_FWK_MODULE(SiStripO2ONoises);

typedef SiStripPayloadAnalyzer<SiStripPedestals> SiStripO2OPedestals;
DEFINE_FWK_MODULE(SiStripO2OPedestals);

typedef SiStripPayloadAnalyzer<SiStripThreshold> SiStripO2OThreshold;
DEFINE_FWK_MODULE(SiStripO2OThreshold);

