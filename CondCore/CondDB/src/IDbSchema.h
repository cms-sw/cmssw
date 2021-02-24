#ifndef CondCore_CondDB_IDbSchema_h
#define CondCore_CondDB_IDbSchema_h

//
#include <boost/date_time/posix_time/posix_time.hpp>

#include "CondFormats/Common/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Binary.h"

namespace cond {

  namespace persistency {

    class ITagTable {
    public:
      virtual ~ITagTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select(const std::string& name) = 0;
      virtual bool select(const std::string& name,
                          cond::TimeType& timeType,
                          std::string& objectType,
                          cond::SynchronizationType& synchronizationType,
                          cond::Time_t& endOfValidity,
                          cond::Time_t& lastValidatedTime,
                          int& protectionCode) = 0;
      virtual bool getMetadata(const std::string& name,
                               std::string& description,
                               boost::posix_time::ptime& insertionTime,
                               boost::posix_time::ptime& modificationTime) = 0;
      virtual void insert(const std::string& name,
                          cond::TimeType timeType,
                          const std::string& objectType,
                          cond::SynchronizationType synchronizationType,
                          cond::Time_t endOfValidity,
                          const std::string& description,
                          cond::Time_t lastValidatedTime,
                          const boost::posix_time::ptime& insertionTime) = 0;
      virtual void update(const std::string& name,
                          cond::SynchronizationType synchronizationType,
                          cond::Time_t& endOfValidity,
                          cond::Time_t lastValidatedTime,
                          const boost::posix_time::ptime& updateTime) = 0;
      virtual void updateMetadata(const std::string& name,
                                  const std::string& description,
                                  const boost::posix_time::ptime& updateTime) = 0;
      virtual void updateValidity(const std::string& name,
                                  cond::Time_t lastValidatedTime,
                                  const boost::posix_time::ptime& updateTime) = 0;
      virtual void setProtectionCode(const std::string& name, int code) = 0;
      virtual void unsetProtectionCode(const std::string& name, int code) = 0;
    };

    class IPayloadTable {
    public:
      virtual ~IPayloadTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select(const cond::Hash& payloadHash,
                          std::string& objectType,
                          cond::Binary& payloadData,
                          cond::Binary& streamerInfoData) = 0;
      virtual bool getType(const cond::Hash& payloadHash, std::string& objectType) = 0;
      virtual cond::Hash insertIfNew(const std::string& objectType,
                                     const cond::Binary& payloadData,
                                     const cond::Binary& streamerInfoData,
                                     const boost::posix_time::ptime& insertionTime) = 0;
    };

    class IIOVTable {
    public:
      virtual ~IIOVTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual size_t getGroups(const std::string& tag,
                               const boost::posix_time::ptime& snapshotTime,
                               unsigned long long groupSize,
                               std::vector<cond::Time_t>& groups) = 0;
      virtual size_t select(const std::string& tag,
                            cond::Time_t lowerGroup,
                            cond::Time_t upperGroup,
                            const boost::posix_time::ptime& snapshotTime,
                            std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) = 0;
      virtual bool getLastIov(const std::string& tag,
                              const boost::posix_time::ptime& snapshotTime,
                              cond::Time_t& since,
                              cond::Hash& hash) = 0;
      virtual bool getSize(const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size) = 0;
      virtual bool getRange(const std::string& tag,
                            cond::Time_t begin,
                            cond::Time_t end,
                            const boost::posix_time::ptime& snapshotTime,
                            std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) = 0;
      virtual void insertOne(const std::string& tag,
                             cond::Time_t since,
                             cond::Hash payloadHash,
                             const boost::posix_time::ptime& insertTime) = 0;
      virtual void insertMany(
          const std::string& tag,
          const std::vector<std::tuple<cond::Time_t, cond::Hash, boost::posix_time::ptime> >& iovs) = 0;
      virtual void eraseOne(const std::string& tag, cond::Time_t since, cond::Hash payloadId) = 0;
      virtual void eraseMany(const std::string& tag,
                             const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) = 0;
      virtual void erase(const std::string& tag) = 0;
    };

    class ITagAccessPermissionTable {
    public:
      virtual ~ITagAccessPermissionTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool getAccessPermission(const std::string& tagName,
                                       const std::string& credential,
                                       int credentialType,
                                       int accessType) = 0;
      virtual void setAccessPermission(const std::string& tagName,
                                       const std::string& credential,
                                       int credentialType,
                                       int accessType) = 0;
      virtual void removeAccessPermission(const std::string& tagName,
                                          const std::string& credential,
                                          int credentialType) = 0;
      virtual void removeEntriesForCredential(const std::string& credential, int credentialType) = 0;
    };

    class ITagLogTable {
    public:
      virtual ~ITagLogTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual void insert(const std::string& tag,
                          const boost::posix_time::ptime& eventTime,
                          const std::string& userName,
                          const std::string& hostName,
                          const std::string& command,
                          const std::string& action,
                          const std::string& userText) = 0;
    };

    class IIOVSchema {
    public:
      virtual ~IIOVSchema() {}
      virtual bool exists() = 0;
      virtual bool create() = 0;
      virtual ITagTable& tagTable() = 0;
      virtual IIOVTable& iovTable() = 0;
      virtual IPayloadTable& payloadTable() = 0;
      virtual ITagLogTable& tagLogTable() = 0;
      virtual ITagAccessPermissionTable& tagAccessPermissionTable() = 0;
    };

    class IGTTable {
    public:
      virtual ~IGTTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select(const std::string& name) = 0;
      virtual bool select(const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime) = 0;
      virtual bool select(const std::string& name,
                          cond::Time_t& validity,
                          std::string& description,
                          std::string& release,
                          boost::posix_time::ptime& snapshotTime) = 0;
      virtual void insert(const std::string& name,
                          cond::Time_t validity,
                          const std::string& description,
                          const std::string& release,
                          const boost::posix_time::ptime& snapshotTime,
                          const boost::posix_time::ptime& insertionTime) = 0;
      virtual void update(const std::string& name,
                          cond::Time_t validity,
                          const std::string& description,
                          const std::string& release,
                          const boost::posix_time::ptime& snapshotTime,
                          const boost::posix_time::ptime& insertionTime) = 0;
    };

    class IGTMapTable {
    public:
      virtual ~IGTMapTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select(const std::string& gtName,
                          std::vector<std::tuple<std::string, std::string, std::string> >& tags) = 0;
      virtual bool select(const std::string& gtName,
                          const std::string& preFix,
                          const std::string& postFix,
                          std::vector<std::tuple<std::string, std::string, std::string> >& tags) = 0;
      virtual void insert(const std::string& gtName,
                          const std::vector<std::tuple<std::string, std::string, std::string> >& tags) = 0;
    };

    class IGTSchema {
    public:
      virtual ~IGTSchema() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual IGTTable& gtTable() = 0;
      virtual IGTMapTable& gtMapTable() = 0;
    };

    class IRunInfoTable {
    public:
      virtual ~IRunInfoTable() {}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select(cond::Time_t runNumber, boost::posix_time::ptime& start, boost::posix_time::ptime& end) = 0;
      virtual cond::Time_t getLastInserted(boost::posix_time::ptime& start, boost::posix_time::ptime& end) = 0;
      virtual bool getInclusiveRunRange(
          cond::Time_t lower,
          cond::Time_t upper,
          std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> >& runData) = 0;
      virtual bool getInclusiveTimeRange(
          const boost::posix_time::ptime& lower,
          const boost::posix_time::ptime& upper,
          std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> >& runData) = 0;
      virtual void insertOne(cond::Time_t runNumber,
                             const boost::posix_time::ptime& start,
                             const boost::posix_time::ptime& end) = 0;
      virtual void insert(
          const std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> >& runs) = 0;
      virtual void updateEnd(cond::Time_t runNumber, const boost::posix_time::ptime& end) = 0;
    };

    class IRunInfoSchema {
    public:
      virtual ~IRunInfoSchema() {}
      virtual bool exists() = 0;
      virtual bool create() = 0;
      virtual IRunInfoTable& runInfoTable() = 0;
    };

  }  // namespace persistency
}  // namespace cond
#endif
