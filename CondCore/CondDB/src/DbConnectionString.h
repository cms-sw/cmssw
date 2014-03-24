#ifndef CondCore_CondDB_DbConnectionString_h
#define CondCore_CondDB_DbConnectionString_h

//
#include <string>

namespace cond {

  namespace persistency {

    std::pair<std::string,std::string> getRealConnectionString( const std::string& initialConnection );

    std::pair<std::string,std::string> getRealConnectionString( const std::string& initialConnection, const std::string& transId );

  }

}

#endif
