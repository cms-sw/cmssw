#ifndef CondTools_DQM_ReadBase_h
#define CondTools_DQM_ReadBase_h

/*
 *  \class ReadBase
 *  
 *  needed for using  coral 
 *  
 *  
*/

#include <string>
//#include "CoralKernel/Context.h"
#include "RelationalAccess/ConnectionService.h"
#include "CoralBase/MessageStream.h"

namespace coral {
  //class IConnection;
  class ISessionProxy;
}

class ReadBase {
 public:
  ReadBase();
  virtual ~ReadBase();
  virtual void run() = 0;
  void setVerbosityLevel( coral::MsgLevel level ) ;
 protected:
  coral::ISessionProxy* connect( const std::string& connectionString,
				 const std::string& user, 
				 const std::string& password );
 private:
  //coral::IConnection* m_connection;
  coral::ConnectionService m_connectionService;
};

#endif
