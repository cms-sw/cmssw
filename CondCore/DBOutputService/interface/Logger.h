#ifndef CONDCORE_DBOUTPUTSERVICE_LOGGER_H
#define CONDCORE_DBOUTPUTSERVICE_LOGGER_H

#include <string>
//#include <iostream>
//
// Package:    
// Class  :     
// 
/**\class 
*/
//
// Author:      Zhen Xie
//
#include "UserLogInfo.h"
namespace coral{
  //class ISchema;  
  class IQuery;
}
namespace cond{
  namespace service{
    class UserLogInfo;
  }
  class CoralTransaction;
  class Connection;
  class SequenceManager;
  class Logger{
  public:
    explicit Logger(Connection* connectionHandle);
    ~Logger();
    bool getWriteLock() throw ();
    bool releaseWriteLock() throw ();
    //NB. for oracle only schema owner can do this 
    void createLogDBIfNonExist();
    //
    //the current local time will be registered as execution time
    // payloadName and containerName are also logged but they are deduced from payloadToken
    void logOperationNow(
			 const cond::service::UserLogInfo& userlogInfo,
			 const std::string& destDB,
			 const std::string& payloadToken,
			 const std::string& iovtag,
			 const std::string& iovtimetype
			 );
    //
    //the current local time will be registered as execution time
    //
    // payloadName and containerName are also logged but they are deduced from payloadToken
    void logFailedOperationNow(
			       const cond::service::UserLogInfo& userlogInfo,
			       const std::string& destDB,
			       const std::string& payloadToken,
			       const std::string& iovtag,
			       const std::string& iovtimetype,
			       const std::string& exceptionMessage
			       );
  private:
    void insertLogRecord(unsigned long long logId,
			const std::string& localtime,
			const std::string& destDB,
			const std::string& payloadToken,
			const cond::service::UserLogInfo& userLogInfo,
			const std::string& iovtag,
			const std::string& iovtimetype,
			const std::string& exceptionMessage);
    
    Connection* m_connectionHandle;
    CoralTransaction& m_coraldb;
    //coral::ISchema& m_schema;
    bool m_locked;
    coral::IQuery* m_statusEditorHandle;
    SequenceManager* m_sequenceManager;
    bool m_logTableExists;
  };
}//ns cond
#endif
