#ifndef CondCore_CondDB_DecodingKey_h
#define CondCore_CondDB_DecodingKey_h

#include <iostream>
#include <string>
#include <set>
#include <map>

namespace cond {

  namespace auth {

    struct ServiceCredentials {
      ServiceCredentials();
      std::string connectionString;
      std::string userName;
      std::string password;
    };

    class KeyGenerator {
    public:
      KeyGenerator();

      std::string make(size_t keySize);
      std::string makeWithRandomSize(size_t maxSize);

    private:
      int m_iteration;
    };

    class DecodingKey {
    public:
      static constexpr const char* const KEY_FMT_VERSION = "2.0";
      static constexpr const char* const FILE_NAME = "db.key";
      static constexpr const char* const FILE_PATH = ".cms_cond/db.key";
      static constexpr size_t DEFAULT_KEY_SIZE = 100;

      static std::string templateFile();

    public:
      DecodingKey();

      virtual ~DecodingKey() {}

      size_t init(const std::string& keyFileName, const std::string& password, bool readMode = true);

      size_t createFromInputFile(const std::string& inputFileName, size_t generatedKeySize = 0);

      void list(std::ostream& out);

      void flush();

      const std::string& version() const;

      const std::string& principalName() const;

      const std::string& principalKey() const;

      bool isNominal() const;

      const std::string& ownerName() const;

      const std::map<std::string, ServiceCredentials>& services() const;

      void addDefaultService(const std::string& connectionString);

      void addService(const std::string& serviceName,
                      const std::string& connectionString,
                      const std::string& userName,
                      const std::string& password);

    private:
      std::string m_fileName;

      std::string m_version;

      bool m_mode;

      std::string m_pwd;

      std::string m_principalName;

      std::string m_principalKey;

      std::string m_owner;

      std::map<std::string, ServiceCredentials> m_services;
    };
  }  // namespace auth
}  // namespace cond

inline cond::auth::KeyGenerator::KeyGenerator() : m_iteration(0) {}

inline cond::auth::ServiceCredentials::ServiceCredentials() : connectionString(""), userName(""), password("") {}

inline cond::auth::DecodingKey::DecodingKey()
    : m_fileName(""),
      m_version(""),
      m_mode(true),
      m_pwd(""),
      m_principalName(""),
      m_principalKey(""),
      m_owner(""),
      m_services() {}

inline const std::string& cond::auth::DecodingKey::version() const { return m_version; }

inline const std::string& cond::auth::DecodingKey::principalName() const { return m_principalName; }

inline const std::string& cond::auth::DecodingKey::principalKey() const { return m_principalKey; }

inline bool cond::auth::DecodingKey::isNominal() const { return !m_owner.empty(); }

inline const std::string& cond::auth::DecodingKey::ownerName() const { return m_owner; }

inline const std::map<std::string, cond::auth::ServiceCredentials>& cond::auth::DecodingKey::services() const {
  return m_services;
}

#endif  //  CondCore_CondDB_DecodingKey_h
