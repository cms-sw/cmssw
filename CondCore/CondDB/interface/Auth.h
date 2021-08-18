#ifndef CondCore_CondDB_Auth_h
#define CondCore_CondDB_Auth_h

#include <string>
#include <set>

namespace cond {

  namespace auth {

    static constexpr const char* const COND_AUTH_PATH = "COND_AUTH_PATH";
    static constexpr const char* const COND_AUTH_SYS = "COND_AUTH_SYS";

    static constexpr const char* const COND_ADMIN_GROUP = "COND_ADMIN_GROUP";

    static constexpr const char* const COND_DEFAULT_ROLE = "COND_DEFAULT_ROLE";
    static constexpr const char* const COND_WRITER_ROLE = "COND_WRITER_ROLE";
    static constexpr const char* const COND_READER_ROLE = "COND_READER_ROLE";
    static constexpr const char* const COND_ADMIN_ROLE = "COND_ADMIN_ROLE";

    static const std::set<std::string> ROLES = {std::string(COND_DEFAULT_ROLE),
                                                std::string(COND_READER_ROLE),
                                                std::string(COND_WRITER_ROLE),
                                                std::string(COND_ADMIN_ROLE)};

    typedef enum { DEFAULT_ROLE = 0, READER_ROLE = 1, WRITER_ROLE = 2, ADMIN_ROLE = 3 } RoleCode;

    static const std::pair<const char*, RoleCode> s_roleCodeArray[] = {std::make_pair(COND_DEFAULT_ROLE, DEFAULT_ROLE),
                                                                       std::make_pair(COND_READER_ROLE, READER_ROLE),
                                                                       std::make_pair(COND_WRITER_ROLE, WRITER_ROLE),
                                                                       std::make_pair(COND_ADMIN_ROLE, ADMIN_ROLE)};

    static constexpr const char* const COND_DEFAULT_PRINCIPAL = "COND_DEFAULT_PRINCIPAL";

    static constexpr const char* const COND_KEY = "Memento";

    static constexpr unsigned int COND_AUTHENTICATION_KEY_SIZE = 30;
    static constexpr unsigned int COND_DB_KEY_SIZE = 30;

    static constexpr size_t COND_SESSION_HASH_SIZE = 16;

    static constexpr int COND_SESSION_HASH_CODE = 4;
    static constexpr int COND_DBKEY_CREDENTIAL_CODE = 1;

    static constexpr int COND_DBTAG_LOCK_ACCESS_CODE = 8;
    static constexpr int COND_DBTAG_WRITE_ACCESS_CODE = 2;
    static constexpr int COND_DBTAG_READ_ACCESS_CODE = 1;
    static constexpr int COND_DBTAG_NO_PROTECTION_CODE = 0;

    static constexpr const char* const COND_AUTH_PATH_PROPERTY = "AuthenticationFile";
  }  // namespace auth

}  // namespace cond
#endif
