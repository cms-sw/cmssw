#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Logger.h"
#include <string>
#include <map>
#include <mutex>

//
// Package:     DBOutputService
// Class  :     PoolDBOutputService
//
/**\class PoolDBOutputService PoolDBOutputService.h CondCore/DBOutputService/interface/PoolDBOutputService.h
   Description: edm service for writing conditions object to DB.  
*/
//
// Author:      Zhen Xie
// Fixes and other changes: Giacomo Govi
//

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm
namespace cond {

  namespace service {

    class PoolDBOutputService {
    public:
      PoolDBOutputService(const edm::ParameterSet& iConfig, edm::ActivityRegistry& iAR);

      static const std::string kSharedResource;

      virtual ~PoolDBOutputService();

      //use these to control connections
      void postEndJob();

      cond::persistency::Session newReadOnlySession(const std::string& connectionString,
                                                    const std::string& transactionId);
      // return the database session in use
      cond::persistency::Session session() const;

      //
      void lockRecords();

      //
      void releaseLocks();

      //
      void startTransaction();
      void commitTransaction();

      //
      std::string tag(const std::string& recordName);
      bool isNewTagRequest(const std::string& recordName);

      template <typename T>
      Hash writeOneIOV(const T& payload, Time_t time, const std::string& recordName) {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        doStartTransaction();
        cond::persistency::TransactionScope scope(m_session.transaction());
        Hash thePayloadHash("");
        try {
          this->initDB();
          auto& myrecord = this->getRecord(recordName);
          m_logger.logInfo() << "Tag mapped to record " << recordName << ": " << myrecord.m_tag;
          bool newTag = isNewTagRequest(recordName);
          if (myrecord.m_onlyAppendUpdatePolicy && !newTag) {
            cond::TagInfo_t tInfo;
            this->getTagInfo(myrecord.m_idName, tInfo);
            cond::Time_t lastSince = tInfo.lastInterval.since;
            if (lastSince == cond::time::MAX_VAL)
              lastSince = 0;
            if (time <= lastSince) {
              m_logger.logInfo() << "Won't append iov with since " << std::to_string(time)
                                 << ", because is less or equal to last available since = " << lastSince;
              if (m_autoCommit)
                doCommitTransaction();
              scope.close();
              return thePayloadHash;
            }
          }
          thePayloadHash = m_session.storePayload(payload);
          std::string payloadType = cond::demangledName(typeid(payload));
          if (newTag) {
            createNewIOV(thePayloadHash, payloadType, time, myrecord);
          } else {
            appendSinceTime(thePayloadHash, time, myrecord);
          }
          if (m_autoCommit) {
            doCommitTransaction();
          }
        } catch (const std::exception& er) {
          cond::throwException(std::string(er.what()), "PoolDBOutputService::writeOne");
        }
        scope.close();
        return thePayloadHash;
      }

      template <typename T>
      void writeMany(const std::map<Time_t, std::shared_ptr<T> >& iovAndPayloads, const std::string& recordName) {
        if (iovAndPayloads.empty())
          return;
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        doStartTransaction();
        cond::persistency::TransactionScope scope(m_session.transaction());
        try {
          this->initDB();
          auto& myrecord = this->getRecord(recordName);
          m_logger.logInfo() << "Tag mapped to record " << recordName << ": " << myrecord.m_tag;
          bool newTag = isNewTagRequest(recordName);
          cond::Time_t lastSince = 0;
          cond::persistency::IOVEditor editor;
          if (newTag) {
            std::string payloadType = cond::demangledName(typeid(T));
            editor = m_session.createIov(payloadType, myrecord.m_tag, myrecord.m_timetype, cond::SYNCH_ANY);
            editor.setDescription("New Tag");
          } else {
            editor = m_session.editIov(myrecord.m_tag);
            if (myrecord.m_onlyAppendUpdatePolicy) {
              cond::TagInfo_t tInfo;
              this->getTagInfo(myrecord.m_idName, tInfo);
              lastSince = tInfo.lastInterval.since;
              if (lastSince == cond::time::MAX_VAL)
                lastSince = 0;
            }
          }
          for (auto& iovEntry : iovAndPayloads) {
            Time_t time = iovEntry.first;
            auto payload = iovEntry.second;
            if (myrecord.m_onlyAppendUpdatePolicy && !newTag) {
              if (time <= lastSince) {
                m_logger.logInfo() << "Won't append iov with since " << std::to_string(time)
                                   << ", because is less or equal to last available since = " << lastSince;
                continue;
              }
            }
            auto payloadHash = m_session.storePayload(*payload);
            editor.insert(time, payloadHash);
          }
          cond::UserLogInfo a = this->lookUpUserLogInfo(myrecord.m_idName);
          editor.flush(a.usertext);
          if (m_autoCommit) {
            doCommitTransaction();
          }
        } catch (const std::exception& er) {
          cond::throwException(std::string(er.what()), "PoolDBOutputService::writeMany");
        }
        scope.close();
        return;
      }

      // close the IOVSequence setting lastTill
      void closeIOV(Time_t lastTill, const std::string& recordName);

      template <typename T>
      void createOneIOV(const T& payload, cond::Time_t firstSinceTime, const std::string& recordName) {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        doStartTransaction();
        cond::persistency::TransactionScope scope(m_session.transaction());
        try {
          this->initDB();
          auto& myrecord = this->getRecord(recordName);
          if (!myrecord.m_isNewTag) {
            cond::throwException(myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
          }
          Hash payloadId = m_session.storePayload(payload);
          createNewIOV(payloadId, cond::demangledName(typeid(payload)), firstSinceTime, myrecord);
          if (m_autoCommit) {
            doCommitTransaction();
          }
        } catch (const std::exception& er) {
          cond::throwException(std::string(er.what()), "PoolDBOutputService::createNewIov");
        }
        scope.close();
      }

      template <typename T>
      void appendOneIOV(const T& payload, cond::Time_t sinceTime, const std::string& recordName) {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        doStartTransaction();
        cond::persistency::TransactionScope scope(m_session.transaction());
        try {
          bool dbexists = this->initDB(true);
          if (!dbexists) {
            cond::throwException(std::string("Target database does not exist."),
                                 "PoolDBOutputService::appendSinceTime");
          }
          auto& myrecord = this->lookUpRecord(recordName);
          if (myrecord.m_isNewTag) {
            cond::throwException(std::string("Cannot append to non-existing tag ") + myrecord.m_tag,
                                 "PoolDBOutputService::appendSinceTime");
          }
          appendSinceTime(m_session.storePayload(payload), sinceTime, myrecord);
          if (m_autoCommit) {
            doCommitTransaction();
          }
        } catch (const std::exception& er) {
          cond::throwException(std::string(er.what()), "PoolDBOutputService::appendSinceTime");
        }
        scope.close();
      }

      void createNewIOV(const std::string& firstPayloadId, cond::Time_t firstSinceTime, const std::string& recordName);

      bool appendSinceTime(const std::string& payloadId, cond::Time_t sinceTime, const std::string& recordName);

      // Remove the payload and its valid sinceTime from the database
      //
      void eraseSinceTime(const std::string& payloadId, cond::Time_t sinceTime, const std::string& recordName);

      //
      // Service time utility method
      // return the infinity value according to the given timetype
      //
      cond::Time_t endOfTime() const;
      //
      // Service time utility method
      // return beginning of time value according to the given timetype
      //
      cond::Time_t beginOfTime() const;
      //
      // Service time utility method
      // return the time value of the current edm::Event according to the
      // given timetype
      //
      cond::Time_t currentTime() const;

      // optional. User can inject additional information into the log associated with a given record
      void setLogHeaderForRecord(const std::string& recordName,
                                 const std::string& provenance,
                                 const std::string& usertext);

      // Retrieve tag information
      bool tagInfo(const std::string& recordName, cond::TagInfo_t& result);

      void forceInit();

      cond::persistency::Logger& logger() { return m_logger; }

      struct Record {
        Record()
            : m_tag(), m_isNewTag(true), m_idName(), m_timetype(cond::runnumber), m_onlyAppendUpdatePolicy(false) {}

        std::string timetypestr() const { return cond::timeTypeSpecs[m_timetype].name; }
        std::string m_tag;
        bool m_isNewTag;
        std::string m_idName;
        cond::TimeType m_timetype;
        unsigned int m_refreshTime = 0;
        bool m_onlyAppendUpdatePolicy;
      };

      const Record& lookUpRecord(const std::string& recordName);

    private:
      //
      void doStartTransaction();
      void doCommitTransaction();

      //
      bool getTagInfo(const std::string& recordName, cond::TagInfo_t& result);

      //
      void createNewIOV(const std::string& firstPayloadId,
                        const std::string payloadType,
                        cond::Time_t firstSinceTime,
                        Record& record);

      // Append the payload and its valid sinceTime into the database
      // Note: the iov index appended to MUST pre-existing and the existing
      // conditions data are retrieved from the DB
      //
      bool appendSinceTime(const std::string& payloadId, cond::Time_t sinceTime, const Record& record);

      //use these to control transaction interval
      void preEventProcessing(edm::StreamContext const&);
      void preGlobalBeginLumi(edm::GlobalContext const&);
      void preGlobalBeginRun(edm::GlobalContext const&);
      void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
      void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);

      void fillRecord(edm::ParameterSet& pset, const std::string& gTimeTypeStr);

      bool initDB(bool readOnly = false);

      Record& getRecord(const std::string& recordName);

      cond::UserLogInfo& lookUpUserLogInfo(const std::string& recordName);

    private:
      cond::persistency::Logger m_logger;
      std::recursive_mutex m_mutex;
      cond::TimeType m_timetype;
      std::vector<cond::Time_t> m_currentTimes;

      cond::persistency::ConnectionPool m_connection;
      cond::persistency::Session m_session;
      bool m_transactionActive;
      bool m_autoCommit;
      unsigned int m_writeTransactionDelay = 0;
      bool m_dbInitialised;

      std::map<std::string, Record> m_records;
      std::map<std::string, cond::UserLogInfo> m_logheaders;

    };  //PoolDBOutputService
  }     // namespace service
}  // namespace cond
#endif
