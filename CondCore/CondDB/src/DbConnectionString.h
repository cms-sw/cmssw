#ifndef CondCore_CondDB_DbConnectionString_h
#define CondCore_CondDB_DbConnectionString_h

#include "CondCore/CondDB/interface/Utils.h"
//
#include <string>

namespace cond {

  namespace persistency {

    std::pair<std::string,std::string> getConnectionParams( const std::string& initialConnection, const std::string& transId );

  }

}

#endif
