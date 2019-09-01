#ifndef CondTools_L1Trigger_DataWriter_h
#define CondTools_L1Trigger_DataWriter_h

// Framework
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/DataKey.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include "DataFormats/Provenance/interface/RunID.h"

// L1T includes
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"

#include "CondTools/L1Trigger/interface/WriterProxy.h"

#include <string>
#include <map>

namespace l1t {

  /* This class is used to write L1 Trigger configuration data to Pool DB.
 * It also has a function for reading L1TriggerKey directly from Pool.
 *
 * In order to use this class to write payloads, user has to make sure to register datatypes that she or he is
 * interested to write to the framework. This should be done with macro REGISTER_L1_WRITER(record, type) found in
 * WriterProxy.h file. Also, one should take care to register these data types to CondDB framework with macro
 * REGISTER_PLUGIN(record, type) from registration_macros.h found in PluginSystem.
 */

  class DataWriterExt {
  public:
    DataWriterExt();
    ~DataWriterExt();

    // Payload and IOV writing functions.

    // Get payload from EventSetup and write to DB with no IOV
    // recordType = "record@type", return value is payload token
    std::string writePayload(const edm::EventSetup& setup, const std::string& recordType);

    // Use PoolDBOutputService to append IOV with sinceRun to IOV sequence
    // for given ESRecord.  PoolDBOutputService knows the corresponding IOV tag.
    // Return value is true if IOV was updated; false if IOV was already
    // up to date.
    bool updateIOV(const std::string& esRecordName,
                   const std::string& payloadToken,
                   edm::RunNumber_t sinceRun,
                   bool logTransactions = false);

    // Write L1TriggerKeyListExt payload and set IOV.  Takes ownership of pointer.
    void writeKeyList(L1TriggerKeyListExt* keyList, edm::RunNumber_t sinceRun = 0, bool logTransactions = false);

    // Read object directly from Pool, not from EventSetup.
    template <class T>
    void readObject(const std::string& payloadToken, T& outputObject);

    std::string payloadToken(const std::string& recordName, edm::RunNumber_t runNumber);

    std::string lastPayloadToken(const std::string& recordName);

    bool fillLastTriggerKeyList(L1TriggerKeyListExt& output);

  protected:
  };

  template <class T>
  void DataWriterExt::readObject(const std::string& payloadToken, T& outputObject) {
    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    poolDb->forceInit();
    cond::persistency::Session session = poolDb->session();
    ///  session.transaction().start(true);

    // Get object from CondDB
    std::shared_ptr<T> ref = session.fetchPayload<T>(payloadToken);
    outputObject = *ref;
    ///  session.transaction().commit ();
  }

}  // namespace l1t

#endif
