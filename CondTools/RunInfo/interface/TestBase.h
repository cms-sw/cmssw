#ifndef CondTools_RunInfo_TestBase_H
#define CondTools_RunInfo_TestBase_H

/*
 *  \class TestBase
 *  
 *  needed for using  coral 
 *  
 *  
*/


#include <string>
#include "CoralKernel/Context.h"
#include "CoralBase/MessageStream.h"

//#include "SealKernel/IMessageService.h"

namespace coral {
  class IConnection;
  class ISession;
}

class TestBase
{
public:
  TestBase();
  virtual ~TestBase();
  virtual void run() = 0;
  void setVerbosityLevel( coral::MsgLevel level ) ;
protected:
  coral::ISession* connect( const std::string& connectionString,
                            const std::string& user, 
                            const std::string& password );

private:
  //seal::Handle<seal::Context> m_context;
  coral::IConnection*         m_connection;
};

#endif
