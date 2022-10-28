#ifndef CondTools_L1Trigger_L1ConfigOnlineProdBase_h
#define CondTools_L1Trigger_L1ConfigOnlineProdBase_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ConfigOnlineProdBase
//
/**\class L1ConfigOnlineProdBase L1ConfigOnlineProdBase.h CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h

 Description: Abstract templated base class for producers that reads OMDS to
 retrieve configuration data for a given key and generates the corresponding
 C++ objects.

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Tue Sep  2 22:48:15 CEST 2008
// $Id: L1ConfigOnlineProdBase.h,v 1.9 2010/02/16 21:55:43 wsun Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "CondTools/L1Trigger/interface/DataWriter.h"
#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include <optional>
// forward declarations

template <class TRcd, class TData>
class L1ConfigOnlineProdBase : public edm::ESProducer {
public:
  L1ConfigOnlineProdBase(const edm::ParameterSet&);
  ~L1ConfigOnlineProdBase() override;

  virtual std::unique_ptr<TData> produce(const TRcd& iRecord);

  virtual std::unique_ptr<TData> newObject(const std::string& objectKey) = 0;

private:
  // ----------member data ---------------------------
  edm::ESGetToken<L1TriggerKeyList, TRcd> l1TriggerKeyListToken_;
  edm::ESGetToken<L1TriggerKey, TRcd> l1TriggerKeyToken_;

protected:
  l1t::OMDSReader m_omdsReader;
  std::optional<edm::ESConsumesCollectorT<TRcd>> m_consumesCollector;
  bool m_forceGeneration;

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
L1ConfigOnlineProdBase<TRcd, TData>::L1ConfigOnlineProdBase(const edm::ParameterSet& iConfig)
    : m_omdsReader(),
      m_forceGeneration(iConfig.getParameter<bool>("forceGeneration")),
      m_dbSession(),
      m_copyFromCondDB(false) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this);

  //now do what ever other initialization is needed
  l1TriggerKeyListToken_ = cc.consumes();
  l1TriggerKeyToken_ = cc.consumes();

  m_consumesCollector = std::move(cc);

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

template <class TRcd, class TData>
L1ConfigOnlineProdBase<TRcd, TData>::~L1ConfigOnlineProdBase() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

template <class TRcd, class TData>
std::unique_ptr<TData> L1ConfigOnlineProdBase<TRcd, TData>::produce(const TRcd& iRecord) {
  std::unique_ptr<TData> pData;

  // Get object key and check if already in ORCON
  std::string key;
  if (getObjectKey(iRecord, key) || m_forceGeneration) {
    if (m_copyFromCondDB) {
      auto keyList = iRecord.getHandle(l1TriggerKeyListToken_);

      // Find payload token
      std::string recordName = edm::typelookup::className<TRcd>();
      std::string dataType = edm::typelookup::className<TData>();
      std::string payloadToken = keyList->token(recordName, dataType, key);

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
      pData = newObject(key);
    }

    //     if( pData.get() == 0 )
    if (pData == std::unique_ptr<TData>()) {
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
bool L1ConfigOnlineProdBase<TRcd, TData>::getObjectKey(const TRcd& iRecord, std::string& objectKey) {
  // Explanation of funny syntax: since record is dependent, we are not
  // expecting getRecord to be a template so the compiler parses it
  // as a non-template. http://gcc.gnu.org/ml/gcc-bugs/2005-11/msg03685.html

  // If L1TriggerKey is invalid, then all configuration objects are
  // already in ORCON.
  edm::ESHandle<L1TriggerKey> key;
  try {
    key = iRecord.getHandle(l1TriggerKeyToken_);
  } catch (l1t::DataAlreadyPresentException& ex) {
    objectKey = std::string();
    return false;
  }

  // Get object key from L1TriggerKey
  std::string recordName = edm::typelookup::className<TRcd>();
  std::string dataType = edm::typelookup::className<TData>();

  objectKey = key->get(recordName, dataType);

  // Get L1TriggerKeyList
  L1TriggerKeyList keyList;
  l1t::DataWriter dataWriter;
  if (!dataWriter.fillLastTriggerKeyList(keyList)) {
    edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyList";
  }

  // If L1TriggerKeyList does not contain object key, token is empty
  return keyList.token(recordName, dataType, objectKey).empty();
}

#endif
