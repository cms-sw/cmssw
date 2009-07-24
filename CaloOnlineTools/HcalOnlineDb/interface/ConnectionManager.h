#ifndef ConnectionManager_hh_included
#define ConnectionManager_hh_included

#include <string>

namespace oracle {
  namespace occi {
    class Connection;
    class Environment;
    class Statement;
  }
}

class ConnectionManager {
 public:
  ConnectionManager();
  bool connect();
  oracle::occi::Statement* getStatement(const std::string& query);
  void disconnect();
 private:
  oracle::occi::Environment *env;
  oracle::occi::Connection *conn;
};

#endif
