#ifndef CONDCORE_DBCOMMON_LOGGER_H
#define CONDCORE_DBCOMMON_LOGGER_H

#include "CondCore/DBCommon/interface/DbSession.h"
#include <string>

//
// Package: CondCore/DBCommon   
// Class  : Logger    
// 
/**\class 
*/
//
// Author:      Zhen Xie
//

namespace coral{
  //class ISchema;  
  class IQuery;
}
namespace cond{
  class UserLogInfo;
  class LogDBEntry;
  class SequenceManager;
  class Logger{
  public:
    explicit Logger(DbSession& sessionHandle);
    ~Logger();

    void connect( const std::string& logConnectionString, bool readOnly=false );
    //NB. for oracle only schema owner can do this 
    void createLogDBIfNonExist();
    //
    //the current local time will be registered as execution time
    // payloadName and containerName are also logged but they are deduced from payloadToken
    void logOperationNow(
			 const cond::UserLogInfo& userlogInfo,
			 const std::string& destDB,
			 const std::string& payloadToken,
                         const std::string& payloadClass,
			 const std::string& iovtag,
			 const std::string& iovtimetype,
			 unsigned int payloadIdx,
			 unsigned long long lastSince
			 );
    //
    //the current local time will be registered as execution time
    //
    // payloadName and containerName are also logged but they are deduced from payloadToken
    void logFailedOperationNow(
			       const cond::UserLogInfo& userlogInfo,
			       const std::string& destDB,
			       const std::string& payloadToken,
                               const std::string& payloadClass,
			       const std::string& iovtag,
			       const std::string& iovtimetype,
			       unsigned int payloadIdx,
			       unsigned long long lastSince,
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
    
    void LookupLastEntryByTag( const std::string& iovtag,
                               const std::string & connectionStr,
			       LogDBEntry& logentry,
			       bool filterFailedOp=true) const;

  private:
    void insertLogRecord(unsigned long long logId,
			 const std::string& utctime,
			 const std::string& destDB,
			 const std::string& payloadToken,
                         const std::string& payloadClass,
			 const cond::UserLogInfo& userLogInfo,
			 const std::string& iovtag,
			 const std::string& iovtimetype,
			 unsigned int payloadIdx,
			 unsigned long long lastSince,
			 const std::string& exceptionMessage);
    
    mutable DbSession m_sessionHandle;
    //coral::ISchema& m_schema;
    bool m_locked;
    coral::IQuery* m_statusEditorHandle;
    SequenceManager* m_sequenceManager;
    bool m_logTableExists;
  };
}//ns cond
#endif
