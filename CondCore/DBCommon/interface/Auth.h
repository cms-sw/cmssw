#ifndef COND_DBCommon_Roles_h
#define COND_DBCommon_Roles_h

#include <string>

namespace cond{

  class Auth {

  public:

    static const char* COND_AUTH_PATH;
    
    static const std::string COND_WRITER_ROLE;
    static const std::string COND_READER_ROLE;
    static const std::string COND_ADMIN_ROLE;
    static const std::string COND_DEFAULT_PRINCIPAL;

    static const std::string COND_KEY;

  };
}

#endif
// DBSESSION_H
