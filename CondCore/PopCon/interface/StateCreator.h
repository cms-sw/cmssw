#ifndef STATECREATOR_H
#define STATECERATOR_H

#include "CondCore/PopCon/interface/DBState.h"
/*#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include <iterator>
*/
//#include <iostream>
#include <string>
//#include <map>
//#include <vector>
namespace cond{
  class DBSession;
  class CoralTransaction;
}
namespace popcon{
  class StateCreator{
  public:
    StateCreator(const std::string& connectionString, 
		 const std::string& offlineString, 
		 const std::string& oname, 
		 bool debug);
    virtual ~StateCreator();
    bool checkAndCompareState();
    bool previousExceptions(bool& fix);
    void setException(std::string ex);
    void generateStatusData();
    void storeStatusData();
  private:
    void getStoredStatusData();
    void getPoolTableName();
    bool compareStatusData();
    
    void initialize();
    void disconnect();
    
    //flag to indicate we're dealing with sqlite destination
    bool m_sqlite;
    
    DBInfo nfo;
    DBState m_saved_state;
    DBState m_current_state;
    
    //popcon metadata connestion string
    std::string m_connect;
    //payload destintion connect string
    std::string m_offline;
    
    bool m_debug;
    
    //Name of the payload class 
    std::string m_obj_name;
    
    cond::DBSession* session;
    cond::CoralTransaction* m_coraldb;
  };
}
#endif
