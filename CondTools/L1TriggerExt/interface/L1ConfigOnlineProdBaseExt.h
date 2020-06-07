#ifndef CondTools_L1TriggerExt_L1ConfigOnlineProdBaseExt_h
#define CondTools_L1TriggerExt_L1ConfigOnlineProdBaseExt_h

// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"
#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"

// forward declarations

template <class TRcd, class TData>
class L1ConfigOnlineProdBaseExt : public edm::ESProducer {
public:
  L1ConfigOnlineProdBaseExt(const edm::ParameterSet&);
  ~L1ConfigOnlineProdBaseExt() override;

  std::unique_ptr<const TData> produce(const TRcd& iRecord);

  virtual std::unique_ptr<const TData> newObject(const std::string& objectKey, const TRcd& iRecord) = 0;

private:
  // ----------member data ---------------------------
  edm::ESGetToken<L1TriggerKeyListExt, L1TriggerKeyListExtRcd> keyList_token;
  edm::ESGetToken<L1TriggerKeyExt, L1TriggerKeyExtRcd> key_token;


protected:
  l1t::OMDSReader m_omdsReader;
  bool m_forceGeneration;
  edm::ESConsumesCollectorT<TRcd> m_setWhatProduced(const edm::ParameterSet&);

  // Called from produce methods.
  // bool is true if the object data should be made.
  // If bool is false, produce method should throw
  // DataAlreadyPresentException.
  bool getObjectKey(const TRcd& record, std::string& objectKey);

  // For reading object directly from a CondDB w/o PoolDBOutputService
  cond::persistency::Session m_dbSession;
  bool m_copyFromCondDB;
};

template <class TRcd, class TData>
L1ConfigOnlineProdBaseExt<TRcd, TData>::L1ConfigOnlineProdBaseExt(const edm::ParameterSet& iConfig)
    : m_omdsReader(),
      m_forceGeneration(iConfig.getParameter<bool>("forceGeneration")),
      m_dbSession(),
      m_copyFromCondDB(false) {
  //the following line is needed to tell the framework what
  // data is being produced
  // setWhatProduced(this)
  //   .setConsumes(keyList_token)
  //   .setConsumes(key_token);

  //now do what ever other initialization is needed

  if (iConfig.exists("copyFromCondDB")) {
    m_copyFromCondDB = iConfig.getParameter<bool>("copyFromCondDB");

    if (m_copyFromCondDB) {
      cond::persistency::ConnectionPool connectionPool;
      // Connect DB Session
      connectionPool.setAuthenticationPath(iConfig.getParameter<std::string>("onlineAuthentication"));
      connectionPool.configure();
      m_dbSession = connectionPool.createSession(iConfig.getParameter<std::string>("onlineDB"));
    }
  } else {
    m_omdsReader.connect(iConfig.getParameter<std::string>("onlineDB"),
                         iConfig.getParameter<std::string>("onlineAuthentication"));
  }
}

template<class TRcd, class TData>
edm::ESConsumesCollectorT<TRcd> L1ConfigOnlineProdBaseExt<TRcd, TData>::m_setWhatProduced(const edm::ParameterSet& iConfig){
  auto collector = setWhatProduced(this);
  collector.setConsumes(keyList_token);
  collector.setConsumes(key_token);
  return collector; 
}


template <class TRcd, class TData>
L1ConfigOnlineProdBaseExt<TRcd, TData>::~L1ConfigOnlineProdBaseExt() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

template <class TRcd, class TData>
std::unique_ptr<const TData> L1ConfigOnlineProdBaseExt<TRcd, TData>::produce(const TRcd& iRecord) {
  std::unique_ptr<const TData> pData;

  // Get object key and check if already in ORCON
  std::string key;
  if (getObjectKey(iRecord, key) || m_forceGeneration) {
    if (m_copyFromCondDB) {
      // Get L1TriggerKeyListExt from EventSetup
      const L1TriggerKeyListExtRcd& keyListRcd =
          ///	 // Get L1TriggerKeyList from EventSetup
          ///	 const L1TriggerKeyListRcd& keyListRcd =
          iRecord.template getRecord<L1TriggerKeyListExtRcd>();
      ///	   iRecord.template getRecord< L1TriggerKeyListRcd >() ;
      ///	 edm::ESHandle< L1TriggerKeyList > keyList ;

      auto const keyList = keyListRcd.get(keyList_token);

      // Find payload token
      std::string recordName = edm::typelookup::className<TRcd>();
      std::string dataType = edm::typelookup::className<TData>();
      std::string payloadToken = keyList.token(recordName, dataType, key);

      edm::LogVerbatim("L1-O2O") << "Copying payload for " << recordName << "@" << dataType << " obj key " << key
                                 << " from CondDB.";
      edm::LogVerbatim("L1-O2O") << "TOKEN " << payloadToken;

      // Get object from POOL
      // Copied from l1t::DataWriter::readObject()
      if (!payloadToken.empty()) {
        m_dbSession.transaction().start();
        pData = m_dbSession.fetchPayload<TData>(payloadToken);
        m_dbSession.transaction().commit();
      }
    } else {
      pData = newObject(key, iRecord);
    }

    //     if( pData.get() == 0 )
    if (pData == std::unique_ptr<const TData>()) {
      std::string dataType = edm::typelookup::className<TData>();

      throw l1t::DataInvalidException("Unable to generate " + dataType + " for key " + key + ".");
    }
  } else {
    std::string dataType = edm::typelookup::className<TData>();

    throw l1t::DataAlreadyPresentException(dataType + " for key " + key + " already in CondDB.");
  }

  return pData;
}

template <class TRcd, class TData>
bool L1ConfigOnlineProdBaseExt<TRcd, TData>::getObjectKey(const TRcd& record, std::string& objectKey) {
  // Get L1TriggerKeyExt
  const L1TriggerKeyExtRcd& keyRcd = record.template getRecord<L1TriggerKeyExtRcd>();

  // Explanation of funny syntax: since record is dependent, we are not
  // expecting getRecord to be a template so the compiler parses it
  // as a non-template. http://gcc.gnu.org/ml/gcc-bugs/2005-11/msg03685.html

  // If L1TriggerKeyExt is invalid, then all configuration objects are
  // already in ORCON.
  // edm::ESHandle<L1TriggerKeyExt> key_token;
  L1TriggerKeyExt key;
  try {
    key = keyRcd.get(key_token);
  } catch (l1t::DataAlreadyPresentException& ex) {
    objectKey = std::string();
    return false;
  }

  // Get object key from L1TriggerKeyExt
  std::string recordName = edm::typelookup::className<TRcd>();
  std::string dataType = edm::typelookup::className<TData>();

  objectKey = key.get(recordName, dataType);

  /*    edm::LogVerbatim( "L1-O2O" ) */
  /*      << "L1ConfigOnlineProdBase record " << recordName */
  /*      << " type " << dataType << " obj key " << objectKey ; */

  // Get L1TriggerKeyListExt
  L1TriggerKeyListExt keyList;
  ///   // Get L1TriggerKeyList
  ///   L1TriggerKeyList keyList ;
  l1t::DataWriterExt dataWriter;
  ///   l1t::DataWriter dataWriter ;
  if (!dataWriter.fillLastTriggerKeyList(keyList)) {
    edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyListExt";
    ///         << "Problem getting last L1TriggerKeyList" ;
  }

  // If L1TriggerKeyList does not contain object key, token is empty

  return keyList.token(recordName, dataType, objectKey).empty();
}

#endif
