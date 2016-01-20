#ifndef COND_DBCommon_Roles_h
#define COND_DBCommon_Roles_h

#include <string>

namespace cond{

  class Auth {

  public:

    static const char* COND_AUTH_PATH;
    static const char* COND_AUTH_SYS;
    
    static const std::string COND_ADMIN_GROUP;

    static const std::string COND_DEFAULT_ROLE;
    static const std::string COND_WRITER_ROLE;
    static const std::string COND_READER_ROLE;
    static const std::string COND_ADMIN_ROLE;
    static const std::string COND_DEFAULT_PRINCIPAL;

    static const std::string COND_KEY;
    static const unsigned int COND_AUTHENTICATION_KEY_SIZE = 30;
    static const unsigned int COND_DB_KEY_SIZE = 30;

    static const std::string COND_AUTH_PATH_PROPERTY;

  };
}

#endif
// DBSESSION_H
