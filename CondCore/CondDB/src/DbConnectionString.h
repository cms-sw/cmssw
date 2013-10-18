#ifndef CondCore_CondDB_DbConnectionString_h
#define CondCore_CondDB_DbConnectionString_h

//
#include <string>

namespace cond {

  namespace persistency {

    std::string getRealConnectionString( const std::string& initialConnection );

    std::string getRealConnectionString( const std::string& initialConnection, const std::string& transId );

  }

}

#endif
