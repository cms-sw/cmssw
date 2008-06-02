#ifndef CONDCORE_DBCOMMON_LOGGER_H
#define CONDCORE_DBCOMMON_LOGGER_H

#include <string>
//#include <iostream>
//
// Package: CondCore/DBCommon   
// Class  : Logger    
// 
/**\class 
*/
//
// Author:      Zhen Xie
//
//#include "UserLogInfo.h"
namespace coral{
  //class ISchema;  
  class IQuery;
}
namespace cond{
  class UserLogInfo;
  class LogDBEntry;
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
			 const cond::UserLogInfo& userlogInfo,
			 const std::string& destDB,
			 const std::string& payloadToken,
			 const std::string& iovtag,
			 const std::string& iovtimetype,
			 unsigned int payloadIdx
			 );
    //
    //the current local time will be registered as execution time
    //
    // payloadName and containerName are also logged but they are deduced from payloadToken
    void logFailedOperationNow(
			       const cond::UserLogInfo& userlogInfo,
			       const std::string& destDB,
			       const std::string& payloadToken,
			       const std::string& iovtag,
			       const std::string& iovtimetype,
			       unsigned int payloadIdx,
			       const std::string& exceptionMessage
			       );
    //
    // Here we query the log for the last entry for these payloads.
    // Parameter  LogDBEntry& result is both input and output
    // As input, it defines query condition. 
    // Last: in the sense of max rowid satisfies the requirement
    // Note: if empty logentry is given, the absolute max is returned which
    // normally is useless. 
    // return empty struct is no result found
    //
    void LookupLastEntryByProvenance( const std::string& provenance,
				      LogDBEntry& logentry,
				      bool filterFailedOp=true) const;
    //
    // Here we query the log for the last entry for these payloads.
    // Parameter  LogDBEntry& result is both input and output
    // As input, it defines query condition. 
    // Last: in the sense of max rowid satisfies the requirement
    // Note: if empty logentry is given, the absolute max is returned which
    // normally is useless.
    //
    void LookupLastEntryByTag( const std::string& iovtag,
			       LogDBEntry& logentry,
			       bool filterFailedOp=true) const;
    
  private:
    void insertLogRecord(unsigned long long logId,
			 const std::string& localtime,
			 const std::string& destDB,
			 const std::string& payloadToken,
			 const cond::UserLogInfo& userLogInfo,
			 const std::string& iovtag,
			 const std::string& iovtimetype,
			 unsigned int payloadIdx,
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
