#ifndef RPCDBCOM_H
#define RPCDBCOM_H

#include <string>
#include "CoralBase/MessageStream.h"

namespace coral {
  class IConnection;
  class ISession;
}

class RPCDBCom
{
 public:
  RPCDBCom();
  virtual ~RPCDBCom();
  virtual void run() = 0;
  void setVerbosityLevel( coral::MsgLevel level );

 protected:
  coral::ISession* connect( const std::string& connectionString,
                            const std::string& userName,
                            const std::string& password );

 private:
  coral::IConnection*         m_connection;
};

#endif

