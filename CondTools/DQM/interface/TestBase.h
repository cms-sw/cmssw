#ifndef CondTools_DQM_TestBase_h
#define CondTools_DQM_TestBase_h

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

namespace coral {
  class IConnection;
  class ISession;
}

class TestBase {
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
  coral::IConnection* m_connection;
};

#endif
