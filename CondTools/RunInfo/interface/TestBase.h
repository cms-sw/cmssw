#ifndef TestBaseRunInfo_H
#define TestBaseRunInfo_H

#include <string>
#include "SealKernel/Context.h"
#include "SealKernel/IMessageService.h"

/*
 *  \class TestBase
 *  
 *  needed for using  coral 
 *  
 *  
*/

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
  void setVerbosityLevel( seal::Msg::Level level );

protected:
  coral::ISession* connect( const std::string& connectionString,
                            const std::string& user, 
                            const std::string& password );

private:
  seal::Handle<seal::Context> m_context;
  coral::IConnection*         m_connection;
};

#endif
