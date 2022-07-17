#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Auth.h"
#include "Utilities/OpenSSL/interface/openssl_init.h"
#include "IOVSchema.h"

namespace cond {

  namespace persistency {

    cond::Hash makeHash(const std::string& objectType, const cond::Binary& data) {
      cms::openssl_init();
      EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
      const EVP_MD* md = EVP_get_digestbyname("SHA1");
      if (!EVP_DigestInit_ex(mdctx, md, nullptr)) {
        throwException("SHA1 initialization error.", "IOVSchema::makeHash");
      }
      if (!EVP_DigestUpdate(mdctx, objectType.c_str(), objectType.size())) {
        throwException("SHA1 processing error (1).", "IOVSchema::makeHash");
      }
      if (!EVP_DigestUpdate(mdctx, data.data(), data.size())) {
        throwException("SHA1 processing error (2).", "IOVSchema::makeHash");
      }
      unsigned char hash[EVP_MAX_MD_SIZE];
      unsigned int md_len = 0;
      if (!EVP_DigestFinal_ex(mdctx, hash, &md_len)) {
        throwException("SHA1 finalization error.", "IOVSchema::makeHash");
      }
      EVP_MD_CTX_free(mdctx);
      char tmp[EVP_MAX_MD_SIZE * 2 + 1];
      // re-write bytes in hex
      if (md_len > 20) {
        md_len = 20;
      }
      for (unsigned int i = 0; i < md_len; i++) {
        ::sprintf(&tmp[i * 2], "%02x", hash[i]);
      }
      tmp[md_len * 2] = 0;
      return tmp;
    }

    TAG::Table::Table(coral::ISchema& schema) : m_schema(schema) {
      if (exists()) {
        std::set<std::string> columns;
        int ncols = m_schema.tableHandle(tname).description().numberOfColumns();
        for (int i = 0; i < ncols; i++)
          columns.insert(m_schema.tableHandle(tname).description().columnDescription(i).name());
        m_isProtectable = columns.count(PROTECTION_CODE::name);
      }
    }

    bool TAG::Table::Table::exists() { return existsTable(m_schema, tname); }

    void TAG::Table::create() {
      if (exists()) {
        throwException("TAG table already exists in this schema.", "TAG::Table::create");
      }
      TableDescription<NAME,
                       TIME_TYPE,
                       OBJECT_TYPE,
                       SYNCHRONIZATION,
                       END_OF_VALIDITY,
                       DESCRIPTION,
                       LAST_VALIDATED_TIME,
                       INSERTION_TIME,
                       MODIFICATION_TIME,
                       PROTECTION_CODE>
          descr(tname);
      descr.setPrimaryKey<NAME>();
      createTable(m_schema, descr.get());
      m_isProtectable = true;
    }

    bool TAG::Table::select(const std::string& name) {
      Query<NAME> q(m_schema);
      q.addCondition<NAME>(name);
      for (auto row : q) {
      }

      return q.retrievedRows();
    }

    bool TAG::Table::select(const std::string& name,
                            cond::TimeType& timeType,
                            std::string& objectType,
                            cond::SynchronizationType& synchronizationType,
                            cond::Time_t& endOfValidity,
                            cond::Time_t& lastValidatedTime,
                            int& protectionCode) {
      if (isProtectable()) {
        Query<TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, LAST_VALIDATED_TIME, PROTECTION_CODE> q(
            m_schema);
        q.addCondition<NAME>(name);
        for (const auto& row : q)
          std::tie(timeType, objectType, synchronizationType, endOfValidity, lastValidatedTime, protectionCode) = row;
        return q.retrievedRows();
      } else {
        Query<TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, LAST_VALIDATED_TIME> q(m_schema);
        q.addCondition<NAME>(name);
        for (const auto& row : q)
          std::tie(timeType, objectType, synchronizationType, endOfValidity, lastValidatedTime) = row;
        protectionCode = 0;

        return q.retrievedRows();
      }
    }

    bool TAG::Table::getMetadata(const std::string& name,
                                 std::string& description,
                                 boost::posix_time::ptime& insertionTime,
                                 boost::posix_time::ptime& modificationTime) {
      Query<DESCRIPTION, INSERTION_TIME, MODIFICATION_TIME> q(m_schema);
      q.addCondition<NAME>(name);
      for (const auto& row : q)
        std::tie(description, insertionTime, modificationTime) = row;
      return q.retrievedRows();
    }

    void TAG::Table::insert(const std::string& name,
                            cond::TimeType timeType,
                            const std::string& objectType,
                            cond::SynchronizationType synchronizationType,
                            cond::Time_t endOfValidity,
                            const std::string& description,
                            cond::Time_t lastValidatedTime,
                            const boost::posix_time::ptime& insertionTime) {
      if (isProtectable()) {
        RowBuffer<NAME,
                  TIME_TYPE,
                  OBJECT_TYPE,
                  SYNCHRONIZATION,
                  END_OF_VALIDITY,
                  DESCRIPTION,
                  LAST_VALIDATED_TIME,
                  INSERTION_TIME,
                  MODIFICATION_TIME,
                  PROTECTION_CODE>
            dataToInsert(std::tie(name,
                                  timeType,
                                  objectType,
                                  synchronizationType,
                                  endOfValidity,
                                  description,
                                  lastValidatedTime,
                                  insertionTime,
                                  insertionTime,
                                  cond::auth::COND_DBTAG_NO_PROTECTION_CODE));
        insertInTable(m_schema, tname, dataToInsert.get());
      } else {
        RowBuffer<NAME,
                  TIME_TYPE,
                  OBJECT_TYPE,
                  SYNCHRONIZATION,
                  END_OF_VALIDITY,
                  DESCRIPTION,
                  LAST_VALIDATED_TIME,
                  INSERTION_TIME,
                  MODIFICATION_TIME>
            dataToInsert(std::tie(name,
                                  timeType,
                                  objectType,
                                  synchronizationType,
                                  endOfValidity,
                                  description,
                                  lastValidatedTime,
                                  insertionTime,
                                  insertionTime));
        insertInTable(m_schema, tname, dataToInsert.get());
      }
    }

    void TAG::Table::update(const std::string& name,
                            cond::SynchronizationType synchronizationType,
                            cond::Time_t& endOfValidity,
                            cond::Time_t lastValidatedTime,
                            const boost::posix_time::ptime& updateTime) {
      UpdateBuffer buffer;
      buffer.setColumnData<SYNCHRONIZATION, END_OF_VALIDITY, LAST_VALIDATED_TIME, MODIFICATION_TIME>(
          std::tie(synchronizationType, endOfValidity, lastValidatedTime, updateTime));
      buffer.addWhereCondition<NAME>(name);
      updateTable(m_schema, tname, buffer);
    }

    void TAG::Table::updateMetadata(const std::string& name,
                                    const std::string& description,
                                    const boost::posix_time::ptime& updateTime) {
      UpdateBuffer buffer;
      buffer.setColumnData<DESCRIPTION, MODIFICATION_TIME>(std::tie(description, updateTime));
      buffer.addWhereCondition<NAME>(name);
      updateTable(m_schema, tname, buffer);
    }

    void TAG::Table::updateValidity(const std::string& name,
                                    cond::Time_t lastValidatedTime,
                                    const boost::posix_time::ptime& updateTime) {
      UpdateBuffer buffer;
      buffer.setColumnData<LAST_VALIDATED_TIME, MODIFICATION_TIME>(std::tie(lastValidatedTime, updateTime));
      buffer.addWhereCondition<NAME>(name);
      updateTable(m_schema, tname, buffer);
    }

    void TAG::Table::setProtectionCode(const std::string& name, int code) {
      if (!isProtectable()) {
        throwException("Tag in this schema are not protectable.", "TAG::Table::create");
      }
      Query<PROTECTION_CODE> q(m_schema);
      q.addCondition<NAME>(name);
      int newCode = 0;
      for (const auto& row : q)
        std::tie(newCode) = row;
      newCode |= code;
      UpdateBuffer buffer;
      buffer.setColumnData<PROTECTION_CODE>(std::tie(newCode));
      buffer.addWhereCondition<NAME>(name);
      updateTable(m_schema, tname, buffer);
    }

    void TAG::Table::unsetProtectionCode(const std::string& name, int code) {
      if (!isProtectable()) {
        throwException("Tag in this schema are not protectable.", "TAG::Table::unsetProtectionCode");
      }
      Query<PROTECTION_CODE> q(m_schema);
      q.addCondition<NAME>(name);
      int presentCode = 0;
      for (const auto& row : q)
        std::tie(presentCode) = row;
      int newCode = presentCode & (~code);
      UpdateBuffer buffer;
      buffer.setColumnData<PROTECTION_CODE>(std::tie(newCode));
      buffer.addWhereCondition<NAME>(name);
      updateTable(m_schema, tname, buffer);
    }

    IOV::Table::Table(coral::ISchema& schema) : m_schema(schema) {}

    bool IOV::Table::exists() { return existsTable(m_schema, tname); }

    void IOV::Table::create() {
      if (exists()) {
        throwException("IOV table already exists in this schema.", "IOV::Schema::create");
      }

      TableDescription<TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME> descr(tname);
      descr.setPrimaryKey<TAG_NAME, SINCE, INSERTION_TIME>();
      descr.setForeignKey<TAG_NAME, TAG::NAME>("TAG_NAME_FK");
      descr.setForeignKey<PAYLOAD_HASH, PAYLOAD::HASH>("PAYLOAD_HASH_FK");
      createTable(m_schema, descr.get());
    }

    size_t IOV::Table::getGroups(const std::string& tag,
                                 const boost::posix_time::ptime& snapshotTime,
                                 unsigned long long gSize,
                                 std::vector<cond::Time_t>& groups) {
      Query<SINCE_GROUP> q(m_schema, true);
      q.addCondition<TAG_NAME>(tag);
      if (!snapshotTime.is_not_a_date_time()) {
        q.addCondition<INSERTION_TIME>(snapshotTime, "<=");
      }
      q.groupBy(SINCE_GROUP::group(gSize));
      q.addOrderClause<SINCE_GROUP>();
      for (auto row : q) {
        groups.push_back(std::get<0>(row));
      }
      return q.retrievedRows();
    }

    size_t IOV::Table::select(const std::string& tag,
                              cond::Time_t lowerSinceGroup,
                              cond::Time_t upperSinceGroup,
                              const boost::posix_time::ptime& snapshotTime,
                              std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) {
      Query<SINCE, PAYLOAD_HASH> q(m_schema);
      q.addCondition<TAG_NAME>(tag);
      if (lowerSinceGroup > 0)
        q.addCondition<SINCE>(lowerSinceGroup, ">=");
      if (upperSinceGroup < cond::time::MAX_VAL)
        q.addCondition<SINCE>(upperSinceGroup, "<");
      if (!snapshotTime.is_not_a_date_time()) {
        q.addCondition<INSERTION_TIME>(snapshotTime, "<=");
      }
      q.addOrderClause<SINCE>();
      q.addOrderClause<INSERTION_TIME>(false);
      size_t initialSize = iovs.size();
      for (auto row : q) {
        // starting from the second iov in the array, skip the rows with older timestamp
        if (iovs.size() - initialSize && std::get<0>(iovs.back()) == std::get<0>(row))
          continue;
        iovs.push_back(row);
      }
      return iovs.size() - initialSize;
    }

    bool IOV::Table::getLastIov(const std::string& tag,
                                const boost::posix_time::ptime& snapshotTime,
                                cond::Time_t& since,
                                cond::Hash& hash) {
      Query<SINCE, PAYLOAD_HASH> q(m_schema);
      q.addCondition<TAG_NAME>(tag);
      if (!snapshotTime.is_not_a_date_time()) {
        q.addCondition<INSERTION_TIME>(snapshotTime, "<=");
      }
      q.addOrderClause<SINCE>(false);
      q.addOrderClause<INSERTION_TIME>(false);
      q.limitReturnedRows(1);
      for (auto row : q) {
        since = std::get<0>(row);
        hash = std::get<1>(row);
        return true;
      }
      return false;
    }

    bool IOV::Table::getSize(const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size) {
      Query<SEQUENCE_SIZE> q(m_schema);
      q.addCondition<TAG_NAME>(tag);
      if (!snapshotTime.is_not_a_date_time()) {
        q.addCondition<INSERTION_TIME>(snapshotTime, "<=");
      }
      for (auto row : q) {
        size = std::get<0>(row);
        return true;
      }
      return false;
    }

    bool IOV::Table::getRange(const std::string& tag,
                              cond::Time_t begin,
                              cond::Time_t end,
                              const boost::posix_time::ptime& snapshotTime,
                              std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) {
      Query<MAX_SINCE> q0(m_schema);
      q0.addCondition<TAG_NAME>(tag);
      q0.addCondition<SINCE>(begin, "<=");
      if (!snapshotTime.is_not_a_date_time()) {
        q0.addCondition<INSERTION_TIME>(snapshotTime, "<=");
      }
      cond::Time_t lower_since = 0;
      for (auto row : q0) {
        lower_since = std::get<0>(row);
      }
      if (!lower_since)
        return false;
      Query<SINCE, PAYLOAD_HASH> q1(m_schema);
      q1.addCondition<TAG_NAME>(tag);
      q1.addCondition<SINCE>(lower_since, ">=");
      if (!snapshotTime.is_not_a_date_time()) {
        q1.addCondition<INSERTION_TIME>(snapshotTime, "<=");
      }
      q1.addCondition<SINCE>(end, "<=");
      q1.addOrderClause<SINCE>();
      q1.addOrderClause<INSERTION_TIME>(false);
      size_t initialSize = iovs.size();
      for (auto row : q1) {
        // starting from the second iov in the array, skip the rows with older timestamp
        if (iovs.size() - initialSize && std::get<0>(iovs.back()) == std::get<0>(row))
          continue;
        iovs.push_back(row);
      }
      return iovs.size() - initialSize;
    }

    void IOV::Table::insertOne(const std::string& tag,
                               cond::Time_t since,
                               cond::Hash payloadHash,
                               const boost::posix_time::ptime& insertTimeStamp) {
      RowBuffer<TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME> dataToInsert(
          std::tie(tag, since, payloadHash, insertTimeStamp));
      insertInTable(m_schema, tname, dataToInsert.get());
    }

    void IOV::Table::insertMany(
        const std::string& tag,
        const std::vector<std::tuple<cond::Time_t, cond::Hash, boost::posix_time::ptime> >& iovs) {
      BulkInserter<TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME> inserter(m_schema, tname);
      for (auto row : iovs)
        inserter.insert(std::tuple_cat(std::tie(tag), row));
      inserter.flush();
    }

    void IOV::Table::eraseOne(const std::string& tag, cond::Time_t since, cond::Hash payloadId) {
      DeleteBuffer buffer;
      buffer.addWhereCondition<TAG_NAME>(tag);
      buffer.addWhereCondition<SINCE>(since);
      buffer.addWhereCondition<PAYLOAD_HASH>(payloadId);
      deleteFromTable(m_schema, tname, buffer);
    }

    void IOV::Table::eraseMany(const std::string& tag, const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) {
      BulkDeleter<TAG_NAME, SINCE, PAYLOAD_HASH> deleter(m_schema, tname);
      for (auto iov : iovs)
        deleter.erase(std::tuple_cat(std::tie(tag), iov));
      deleter.flush();
    }

    void IOV::Table::erase(const std::string& tag) {
      DeleteBuffer buffer;
      buffer.addWhereCondition<TAG_NAME>(tag);
      deleteFromTable(m_schema, tname, buffer);
    }

    TAG_LOG::Table::Table(coral::ISchema& schema) : m_schema(schema) {}

    bool TAG_LOG::Table::exists() { return existsTable(m_schema, tname); }

    void TAG_LOG::Table::create() {
      if (exists()) {
        throwException("TAG_LOG table already exists in this schema.", "TAG_LOG::create");
      }
      TableDescription<TAG_NAME, EVENT_TIME, USER_NAME, HOST_NAME, COMMAND, ACTION, USER_TEXT> descr(tname);
      descr.setPrimaryKey<TAG_NAME, EVENT_TIME, ACTION>();
      descr.setForeignKey<TAG_NAME, TAG::NAME>("TAG_NAME_FK");
      createTable(m_schema, descr.get());
    }

    void TAG_LOG::Table::insert(const std::string& tag,
                                const boost::posix_time::ptime& eventTime,
                                const std::string& userName,
                                const std::string& hostName,
                                const std::string& command,
                                const std::string& action,
                                const std::string& userText) {
      RowBuffer<TAG_NAME, EVENT_TIME, USER_NAME, HOST_NAME, COMMAND, ACTION, USER_TEXT> dataToInsert(
          std::tie(tag, eventTime, userName, hostName, command, action, userText));
      insertInTable(m_schema, tname, dataToInsert.get());
    }

    TAG_AUTHORIZATION::Table::Table(coral::ISchema& schema) : m_schema(schema) {}

    bool TAG_AUTHORIZATION::Table::exists() { return existsTable(m_schema, tname); }

    void TAG_AUTHORIZATION::Table::create() {
      if (exists()) {
        throwException("TAG_AUTHORIZATION table already exists in this schema.", "TAG_AUTHORIZATION::create");
      }
      TableDescription<TAG_NAME, ACCESS_TYPE, CREDENTIAL, CREDENTIAL_TYPE> descr(tname);
      descr.setPrimaryKey<TAG_NAME, CREDENTIAL, CREDENTIAL_TYPE>();
      descr.setForeignKey<TAG_NAME, TAG::NAME>("TAG_NAME_FK");
      createTable(m_schema, descr.get());
    }

    bool TAG_AUTHORIZATION::Table::getAccessPermission(const std::string& tagName,
                                                       const std::string& credential,
                                                       int credentialType,
                                                       int accessType) {
      Query<ACCESS_TYPE> q(m_schema);
      q.addCondition<TAG_NAME>(tagName);
      q.addCondition<CREDENTIAL>(credential);
      q.addCondition<CREDENTIAL_TYPE>(credentialType);
      int allowedAccess = 0;
      for (auto row : q) {
        allowedAccess = std::get<0>(row);
      }
      return allowedAccess & accessType;
    }

    void TAG_AUTHORIZATION::Table::setAccessPermission(const std::string& tagName,
                                                       const std::string& credential,
                                                       int credentialType,
                                                       int accessType) {
      RowBuffer<TAG_NAME, ACCESS_TYPE, CREDENTIAL, CREDENTIAL_TYPE> dataToInsert(
          std::tie(tagName, accessType, credential, credentialType));
      insertInTable(m_schema, tname, dataToInsert.get());
    }

    void TAG_AUTHORIZATION::Table::removeAccessPermission(const std::string& tagName,
                                                          const std::string& credential,
                                                          int credentialType) {
      DeleteBuffer buffer;
      buffer.addWhereCondition<TAG_NAME>(tagName);
      buffer.addWhereCondition<CREDENTIAL>(credential);
      buffer.addWhereCondition<CREDENTIAL_TYPE>(credentialType);
      deleteFromTable(m_schema, tname, buffer);
    }

    void TAG_AUTHORIZATION::Table::removeEntriesForCredential(const std::string& credential, int credentialType) {
      DeleteBuffer buffer;
      buffer.addWhereCondition<CREDENTIAL>(credential);
      buffer.addWhereCondition<CREDENTIAL_TYPE>(credentialType);
      deleteFromTable(m_schema, tname, buffer);
    }

    PAYLOAD::Table::Table(coral::ISchema& schema) : m_schema(schema) {}

    bool PAYLOAD::Table::exists() { return existsTable(m_schema, tname); }

    void PAYLOAD::Table::create() {
      if (exists()) {
        throwException("Payload table already exists in this schema.", "PAYLOAD::Schema::create");
      }

      TableDescription<HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME> descr(tname);
      descr.setPrimaryKey<HASH>();
      createTable(m_schema, descr.get());
    }

    bool PAYLOAD::Table::select(const cond::Hash& payloadHash) {
      Query<HASH> q(m_schema);
      q.addCondition<HASH>(payloadHash);
      for (auto row : q) {
      }

      return q.retrievedRows();
    }

    bool PAYLOAD::Table::getType(const cond::Hash& payloadHash, std::string& objectType) {
      Query<OBJECT_TYPE> q(m_schema);
      q.addCondition<HASH>(payloadHash);
      for (auto row : q) {
        objectType = std::get<0>(row);
      }

      return q.retrievedRows();
    }

    bool PAYLOAD::Table::select(const cond::Hash& payloadHash,
                                std::string& objectType,
                                cond::Binary& payloadData,
                                cond::Binary& streamerInfoData) {
      Query<DATA, STREAMER_INFO, OBJECT_TYPE> q(m_schema);
      q.addCondition<HASH>(payloadHash);
      for (const auto& row : q) {
        std::tie(payloadData, streamerInfoData, objectType) = row;
      }
      return q.retrievedRows();
    }

    bool PAYLOAD::Table::insert(const cond::Hash& payloadHash,
                                const std::string& objectType,
                                const cond::Binary& payloadData,
                                const cond::Binary& streamerInfoData,
                                const boost::posix_time::ptime& insertionTime) {
      std::string version("dummy");
      cond::Binary sinfoData(streamerInfoData);
      if (!sinfoData.size())
        sinfoData.copy(std::string("0"));
      RowBuffer<HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME> dataToInsert(
          std::tie(payloadHash, objectType, payloadData, sinfoData, version, insertionTime));
      bool failOnDuplicate = false;
      return insertInTable(m_schema, tname, dataToInsert.get(), failOnDuplicate);
    }

    cond::Hash PAYLOAD::Table::insertIfNew(const std::string& payloadObjectType,
                                           const cond::Binary& payloadData,
                                           const cond::Binary& streamerInfoData,
                                           const boost::posix_time::ptime& insertionTime) {
      cond::Hash payloadHash = makeHash(payloadObjectType, payloadData);
      // the check on the hash existance is only required to avoid the error message printing in SQLite! once this is removed, this check is useless...
      if (!select(payloadHash)) {
        insert(payloadHash, payloadObjectType, payloadData, streamerInfoData, insertionTime);
      }
      return payloadHash;
    }

    IOVSchema::IOVSchema(coral::ISchema& schema)
        : m_tagTable(schema),
          m_iovTable(schema),
          m_tagLogTable(schema),
          m_tagAccessPermissionTable(schema),
          m_payloadTable(schema) {}

    bool IOVSchema::exists() {
      if (!m_tagTable.exists())
        return false;
      if (!m_payloadTable.exists())
        return false;
      if (!m_iovTable.exists())
        return false;
      return true;
    }

    bool IOVSchema::create() {
      bool created = false;
      if (!exists()) {
        m_tagTable.create();
        m_payloadTable.create();
        m_iovTable.create();
        m_tagLogTable.create();
        m_tagAccessPermissionTable.create();
        created = true;
      }
      return created;
    }

    ITagTable& IOVSchema::tagTable() { return m_tagTable; }

    IIOVTable& IOVSchema::iovTable() { return m_iovTable; }

    ITagLogTable& IOVSchema::tagLogTable() { return m_tagLogTable; }

    ITagAccessPermissionTable& IOVSchema::tagAccessPermissionTable() {
      if (!m_tagTable.isProtectable()) {
        throwException("Tag in this schema are not protectable.", "IOVSchema::tagAccessPermissionTable");
      }
      return m_tagAccessPermissionTable;
    }

    IPayloadTable& IOVSchema::payloadTable() { return m_payloadTable; }

  }  // namespace persistency
}  // namespace cond
