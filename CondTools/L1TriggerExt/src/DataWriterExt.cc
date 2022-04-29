#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"
#include "CondTools/L1Trigger/interface/Exception.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "CondCore/CondDB/interface/Serialization.h"

#include <utility>
#include <typeinfo>

namespace l1t {

  // Default Constructor
  DataWriterExt::DataWriterExt() {}

  // Constructor Needed for Writers with Rcds dependednt on Online Producers
  // they need to be constructed at "construction time" to register what they produce
  DataWriterExt::DataWriterExt(const std::string& recordType) {
    WriterFactory* factory = WriterFactory::get();
    writer_ = factory->create(recordType + "@Writer");
    if (writer_.get() == nullptr) {
      throw cond::Exception("DataWriter: could not create WriterProxy with name " + recordType + "@Writer");
    }
    edm::LogVerbatim("L1-O2O DataWriterExt::DataWriterExt")
        << "Created new " << typeid(writer_).name() << "  | address " << writer_.get() << "  | rcd " << recordType;
  }

  DataWriterExt::~DataWriterExt() {}

  std::string DataWriterExt::writePayload(const edm::EventSetup& setup) {
    edm::LogVerbatim("L1-O2O DataWriterExt::writePayload") << "Will use stored writer at " << writer_.get();

    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    // 2010-02-16: Move session and transaction to WriterProxy::save().  Otherwise, if another transaction is
    // started while WriterProxy::save() is called (e.g. in a ESProducer like L1ConfigOnlineProdBase), the
    // transaction here will become read-only.
    std::string payloadToken = writer_->save(setup);
    auto& writerRef = *writer_.get();
    edm::LogVerbatim("L1-O2O") << typeid(writerRef).name() << " PAYLOAD TOKEN " << payloadToken;

    return payloadToken;
  }

  std::string DataWriterExt::writePayload(const edm::EventSetup& setup, const std::string& recordType) {
    WriterFactory* factory = WriterFactory::get();
    WriterProxyPtr writer(factory->create(recordType + "@Writer"));
    if (writer.get() == nullptr) {
      throw cond::Exception("DataWriter: could not create WriterProxy with name " + recordType + "@Writer");
    }
    writer_ = std::move(writer);

    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    // 2010-02-16: Move session and transaction to WriterProxy::save().  Otherwise, if another transaction is
    // started while WriterProxy::save() is called (e.g. in a ESProducer like L1ConfigOnlineProdBase), the
    // transaction here will become read-only.
    //   cond::DbSession session = poolDb->session();
    //   cond::DbScopedTransaction tr(session);

    ///  cond::persistency::TransactionScope tr(poolDb->session().transaction());
    //   // if throw transaction will unroll
    //   tr.start(false);

    // update key to have new payload registered for record-type pair.
    //  std::string payloadToken = writer->save( setup, session ) ;
    std::string payloadToken = writer_->save(setup);
    auto& writerRef = *writer_.get();
    edm::LogVerbatim("L1-O2O") << typeid(writerRef).name() << " PAYLOAD TOKEN " << payloadToken;

    ////  tr.close();
    //   tr.commit ();

    return payloadToken;
  }

  void DataWriterExt::writeKeyList(L1TriggerKeyListExt* keyList, edm::RunNumber_t sinceRun, bool logTransactions) {
    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    poolDb->forceInit();
    cond::persistency::Session session = poolDb->session();
    if (not session.transaction().isActive())
      session.transaction().start(false);

    // Write L1TriggerKeyListExt payload and save payload token before committing
    std::shared_ptr<L1TriggerKeyListExt> pointer(keyList);
    std::string payloadToken = session.storePayload(*pointer);

    // Commit before calling updateIOV(), otherwise PoolDBOutputService gets
    // confused. ??? why?

    // Set L1TriggerKeyListExt IOV
    session.transaction().commit();
    updateIOV("L1TriggerKeyListExtRcd", payloadToken, sinceRun, logTransactions);
  }

  bool DataWriterExt::updateIOV(const std::string& esRecordName,
                                const std::string& payloadToken,
                                edm::RunNumber_t sinceRun,
                                bool logTransactions) {
    edm::LogVerbatim("L1-O2O") << esRecordName << " PAYLOAD TOKEN " << payloadToken;

    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    bool iovUpdated = true;

    if (poolDb->isNewTagRequest(esRecordName)) {
      sinceRun = poolDb->beginOfTime();
      poolDb->createNewIOV(payloadToken, sinceRun, esRecordName);
    } else {
      cond::TagInfo_t tagInfo;
      poolDb->tagInfo(esRecordName, tagInfo);

      if (sinceRun == 0)  // find last since and add 1
      {
        sinceRun = tagInfo.lastInterval.since;
        ++sinceRun;
      }

      if (tagInfo.lastInterval.payloadId != payloadToken) {
        poolDb->appendSinceTime(payloadToken, sinceRun, esRecordName);
      } else {
        iovUpdated = false;
        edm::LogVerbatim("L1-O2O") << "IOV already up to date.";
      }
    }

    if (iovUpdated) {
      edm::LogVerbatim("L1-O2O") << esRecordName << " " << poolDb->tag(esRecordName) << " SINCE " << sinceRun;
    }

    return iovUpdated;
  }

  std::string DataWriterExt::payloadToken(const std::string& recordName, edm::RunNumber_t runNumber) {
    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    // Get tag corresponding to EventSetup record name.
    std::string iovTag = poolDb->tag(recordName);

    // Get IOV token for tag.
    cond::persistency::Session session = poolDb->session();
    cond::persistency::IOVProxy iov = session.readIov(iovTag);
    session.transaction().start();
    auto iovs = iov.selectAll();
    std::string payloadToken("");
    auto iP = iovs.find(runNumber);
    if (iP != iovs.end()) {
      payloadToken = (*iP).payloadId;
    }
    session.transaction().commit();
    return payloadToken;
  }

  std::string DataWriterExt::lastPayloadToken(const std::string& recordName) {
    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (!poolDb.isAvailable()) {
      throw cond::Exception("DataWriter: PoolDBOutputService not available.");
    }

    cond::TagInfo_t tagInfo;
    poolDb->tagInfo(recordName, tagInfo);
    return tagInfo.lastInterval.payloadId;
  }

  bool DataWriterExt::fillLastTriggerKeyList(L1TriggerKeyListExt& output) {
    std::string keyListToken = lastPayloadToken("L1TriggerKeyListExtRcd");
    if (keyListToken.empty()) {
      return false;
    } else {
      readObject(keyListToken, output);
      return true;
    }
  }

}  // namespace l1t
