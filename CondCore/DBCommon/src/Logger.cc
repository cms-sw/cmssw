#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/SequenceManager.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/ORA/interface/PoolToken.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "LogDBNames.h"
#include <boost/date_time/posix_time/posix_time_types.hpp> //no i/o just types

#include <sstream>
#include <exception>
namespace cond{
  template <class T> 
  std::string to_string(const T& t){
    std::stringstream ss;
    ss<<t;
    return ss.str();
  }
}
cond::Logger::Logger(cond::DbSession& sessionHandle):m_sessionHandle(sessionHandle),m_locked(false),m_statusEditorHandle(0),m_sequenceManager(0),m_logTableExists(false){
}

void 
cond::Logger::createLogDBIfNonExist(){
  if(m_logTableExists) return;
  cond::DbScopedTransaction trans( m_sessionHandle );
  trans.start(false);
  if(m_sessionHandle.nominalSchema().existsTable(cond::LogDBNames::SequenceTableName()) &&
     m_sessionHandle.nominalSchema().existsTable(cond::LogDBNames::LogTableName())){
    m_logTableExists=true;
    trans.commit();
    return;
  }
  //create sequence table
  cond::SequenceManager sequenceGenerator(m_sessionHandle,cond::LogDBNames::SequenceTableName());
  if( !sequenceGenerator.existSequencesTable() ){
    sequenceGenerator.createSequencesTable();
  }
  //create log table
  coral::TableDescription description( "CONDLOG" );
  description.setName(cond::LogDBNames::LogTableName());
  description.insertColumn(std::string("LOGID"),
			   coral::AttributeSpecification::typeNameForType<unsigned long long>() );
  description.setPrimaryKey( std::vector<std::string>( 1, std::string("LOGID")));
  description.insertColumn(std::string("EXECTIME"),
			   coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("EXECTIME"));
  
  description.insertColumn(std::string("IOVTAG"),
			   coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("IOVTAG"));

  description.insertColumn(std::string("IOVTIMETYPE"),
			   coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("IOVTIMETYPE"));

  description.insertColumn(std::string("PAYLOADCLASS"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("PAYLOADCLASS"));

  description.insertColumn(std::string("PAYLOADTOKEN"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("PAYLOADTOKEN"));

  description.insertColumn(std::string("PAYLOADINDEX"),
	  coral::AttributeSpecification::typeNameForType<unsigned int>() );
  description.setNotNullConstraint(std::string("PAYLOADINDEX"));

  description.insertColumn(std::string("LASTSINCE"),
	  coral::AttributeSpecification::typeNameForType<unsigned long long>() );
  description.setNotNullConstraint(std::string("LASTSINCE"));

  description.insertColumn(std::string("DESTINATIONDB"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("DESTINATIONDB"));

  description.insertColumn(std::string("PROVENANCE"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.insertColumn(std::string("USERTEXT"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.insertColumn(std::string("EXECMESSAGE"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  m_sessionHandle.nominalSchema().createTable( description );
  m_logTableExists=true;
  trans.commit();
}
void 
cond::Logger::logOperationNow(
			      const cond::UserLogInfo& userlogInfo,
			      const std::string& destDB,
                              const std::string& payloadClass,
			      const std::string& payloadToken,
			      const std::string& iovtag,
			      const std::string& iovtimetype,
			      unsigned int payloadIdx,
			      unsigned long long lastSince
			      ){
  cond::DbScopedTransaction trans( m_sessionHandle );
  trans.start(false);
  //aquireutctime
  //using namespace boost::posix_time;
  boost::posix_time::ptime p=boost::posix_time::microsec_clock::universal_time();
  std::string now=cond::to_string(p.date().year())+"-"+cond::to_string(p.date().month())+"-"+cond::to_string(p.date().day())+"-"+cond::to_string(p.time_of_day().hours())+":"+cond::to_string(p.time_of_day().minutes())+":"+cond::to_string(p.time_of_day().seconds());
  //aquireentryid
  if(!m_sequenceManager){
    m_sequenceManager=new cond::SequenceManager(m_sessionHandle,cond::LogDBNames::SequenceTableName());
  }
  unsigned long long targetLogId=m_sequenceManager->incrementId(LogDBNames::LogTableName());
  //insert log record with the new id
  this->insertLogRecord(targetLogId,now,destDB,payloadClass,payloadToken,userlogInfo,iovtag,iovtimetype,payloadIdx,lastSince,"OK");
  trans.commit();
}
void 
cond::Logger::logFailedOperationNow(
			       const cond::UserLogInfo& userlogInfo,
			       const std::string& destDB,
                               const std::string& payloadClass,
			       const std::string& payloadToken,
			       const std::string& iovtag,
			       const std::string& iovtimetype,
			       unsigned int payloadIdx,
			       unsigned long long lastSince,
			       const std::string& exceptionMessage
				    ){
  cond::DbScopedTransaction trans( m_sessionHandle );
  trans.start(false);
  //aquireutctime
  boost::posix_time::ptime p=boost::posix_time::microsec_clock::universal_time();
  std::string now=cond::to_string(p.date().year())+"-"+cond::to_string(p.date().month())+"-"+cond::to_string(p.date().day())+"-"+cond::to_string(p.time_of_day().hours())+":"+cond::to_string(p.time_of_day().minutes())+":"+cond::to_string(p.time_of_day().seconds());
  //aquireentryid
  if(!m_sequenceManager){
    m_sequenceManager=new cond::SequenceManager(m_sessionHandle,cond::LogDBNames::SequenceTableName());
  }
  unsigned long long targetLogId=m_sequenceManager->incrementId(LogDBNames::LogTableName());
  //insert log record with the new id
  this->insertLogRecord(targetLogId,now,destDB,payloadClass,payloadToken,userlogInfo,iovtag,iovtimetype,payloadIdx,lastSince,exceptionMessage);
  trans.commit();
}

void 
cond::Logger::LookupLastEntryByProvenance(const std::string& provenance,
					  LogDBEntry& logentry,
					  bool filterFailedOp) const{
  //construct where
  std::string whereClause=cond::LogDBNames::LogTableName();
  whereClause+=".PROVENANCE=:provenance";
  if(filterFailedOp){
    whereClause+=std::string(" AND ");
    whereClause+=cond::LogDBNames::LogTableName();
    whereClause+=std::string(".EXECMESSAGE=:execmessage");
  }
  coral::AttributeList BindVariableList;
  BindVariableList.extend("provenance",typeid(std::string) );
  BindVariableList.extend("execmessage",typeid(std::string) );
  BindVariableList["provenance"].data<std::string>()=provenance;
  BindVariableList["execmessage"].data<std::string>()="OK";
  m_sessionHandle.transaction().start(true);
  {
    std::auto_ptr<coral::IQuery> query(m_sessionHandle.nominalSchema().tableHandle( cond::LogDBNames::LogTableName() ).newQuery());
    // construct select
    query->addToOutputList( "LOGID" );
    query->defineOutputType( "LOGID", "unsigned long long" );
    query->addToOutputList( "DESTINATIONDB" );
    query->addToOutputList( "PROVENANCE" );
    query->addToOutputList( "USERTEXT" );
    query->addToOutputList( "IOVTAG" );
    query->addToOutputList( "IOVTIMETYPE" );
    query->addToOutputList( "PAYLOADINDEX" );
    query->defineOutputType( "PAYLOADINDEX", "unsigned int" );
    query->addToOutputList( "LASTSINCE" );
    query->defineOutputType( "LASTSINCE", "unsigned long long" );
    query->addToOutputList( "PAYLOADCLASS" );
    query->addToOutputList( "PAYLOADTOKEN" );
    query->addToOutputList( "EXECTIME" );
    query->addToOutputList( "EXECMESSAGE" );
    
    query->setCondition( whereClause, BindVariableList );
    query->addToOrderList( cond::LogDBNames::LogTableName()+".LOGID desc" );
    query->limitReturnedRows();
    coral::ICursor& cursor = query->execute();
    if( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      logentry.logId=row["LOGID"].data<unsigned long long>();
      logentry.destinationDB=row["DESTINATIONDB"].data<std::string>();
      logentry.provenance=row["PROVENANCE"].data<std::string>();
      logentry.usertext=row["USERTEXT"].data<std::string>();
      logentry.iovtag=row["IOVTAG"].data<std::string>();
      logentry.iovtimetype=row["IOVTIMETYPE"].data<std::string>();
      logentry.payloadIdx=row["PAYLOADINDEX"].data<unsigned int>();
      logentry.lastSince=row["LASTSINCE"].data<unsigned long long>();
      logentry.payloadClass=row["PAYLOADCLASS"].data<std::string>();
      logentry.payloadToken=row["PAYLOADTOKEN"].data<std::string>();
      logentry.exectime=row["EXECTIME"].data<std::string>();
      logentry.execmessage=row["EXECMESSAGE"].data<std::string>();
    }
  }
  m_sessionHandle.transaction().commit();
}
void 
cond::Logger::LookupLastEntryByTag( const std::string& iovtag,
				    const std::string & connectionStr,
				    cond::LogDBEntry& logentry,
				    bool filterFailedOp) const{
  coral::AttributeList BindVariableList;
  BindVariableList.extend("IOVTAG",typeid(std::string) );
  BindVariableList["IOVTAG"].data<std::string>()=iovtag;
  std::string whereClause("");
  whereClause+=std::string("IOVTAG=:IOVTAG");
  if(connectionStr!=""){
    whereClause+=std::string(" AND ");
    whereClause+=std::string("DESTINATIONDB=:DESTINATIONDB");
    BindVariableList.extend("DESTINATIONDB",typeid(std::string) ); 
    BindVariableList["DESTINATIONDB"].data<std::string>()=connectionStr;
  }   
  if(filterFailedOp){
    whereClause+=std::string(" AND ");
    whereClause+=std::string("EXECMESSAGE=:EXECMESSAGE");
    BindVariableList.extend("EXECMESSAGE",typeid(std::string) );
    BindVariableList["EXECMESSAGE"].data<std::string>()="OK"; 
  }
  m_sessionHandle.transaction().start(true);
  {
    std::auto_ptr<coral::IQuery> query( m_sessionHandle.nominalSchema().tableHandle(cond::LogDBNames::LogTableName()).newQuery() );
    // construct select
    query->addToOutputList( "LOGID" );
    query->defineOutputType( "LOGID", "unsigned long long" );
    query->addToOutputList( "DESTINATIONDB" );
    query->addToOutputList( "PROVENANCE" );
    query->addToOutputList( "USERTEXT" );
    query->addToOutputList( "IOVTAG" );
    query->addToOutputList( "IOVTIMETYPE" );
    query->addToOutputList( "PAYLOADINDEX" );
    query->defineOutputType( "PAYLOADINDEX", "unsigned int" );
    query->addToOutputList( "LASTSINCE" );
    query->defineOutputType( "LASTSINCE", "unsigned long long" );
    query->addToOutputList( "PAYLOADCLASS" );
    query->addToOutputList( "PAYLOADTOKEN" );
    query->addToOutputList( "EXECTIME" );
    query->addToOutputList( "EXECMESSAGE" );
    
    query->setCondition( whereClause, BindVariableList );
    query->addToOrderList( "LOGID DESC" );
    query->limitReturnedRows();
    coral::ICursor& cursor = query->execute();
    if( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      logentry.logId=row["LOGID"].data<unsigned long long>();
      logentry.destinationDB=row["DESTINATIONDB"].data<std::string>();
      logentry.provenance=row["PROVENANCE"].data<std::string>();
      logentry.usertext=row["USERTEXT"].data<std::string>();
      logentry.iovtag=row["IOVTAG"].data<std::string>();
      logentry.iovtimetype=row["IOVTIMETYPE"].data<std::string>();
      logentry.payloadIdx=row["PAYLOADINDEX"].data<unsigned int>();
      logentry.lastSince=row["LASTSINCE"].data<unsigned long long>();
      logentry.payloadClass=row["PAYLOADCLASS"].data<std::string>();
      logentry.payloadToken=row["PAYLOADTOKEN"].data<std::string>();
      logentry.exectime=row["EXECTIME"].data<std::string>();
      logentry.execmessage=row["EXECMESSAGE"].data<std::string>();
      
    }
  }  
  m_sessionHandle.transaction().commit();
}
void 
cond::Logger::LookupLastEntryByTag( const std::string& iovtag,
				    LogDBEntry& logentry,
				    bool filterFailedOp ) const{
  LookupLastEntryByTag(iovtag,"",logentry,filterFailedOp);
}
void
cond::Logger::insertLogRecord(unsigned long long logId,
			      const std::string& utctime,
			      const std::string& destDB,
                              const std::string& payloadClass,
			      const std::string& payloadToken,
			      const cond::UserLogInfo& userLogInfo,
			      const std::string& iovtag,
			      const std::string& iovtimetype,
			      unsigned int payloadIdx,
			      unsigned long long  lastSince,
			      const std::string& exceptionMessage){
  try{
    coral::AttributeList rowData;
    rowData.extend<unsigned long long>("LOGID");
    rowData.extend<std::string>("EXECTIME");
    rowData.extend<std::string>("DESTINATIONDB");
    rowData.extend<std::string>("PAYLOADCLASS");
    rowData.extend<std::string>("PAYLOADTOKEN");
    rowData.extend<std::string>("PROVENANCE");
    rowData.extend<std::string>("USERTEXT");
    rowData.extend<std::string>("IOVTAG");
    rowData.extend<std::string>("IOVTIMETYPE");
    rowData.extend<unsigned int>("PAYLOADINDEX");
    rowData.extend<unsigned long long>("LASTSINCE");
    rowData.extend<std::string>("EXECMESSAGE");
    rowData["LOGID"].data< unsigned long long >() = logId;
    rowData["EXECTIME"].data< std::string >() = utctime;
    rowData["DESTINATIONDB"].data< std::string >() = destDB;
    rowData["PAYLOADCLASS"].data< std::string >() = payloadClass;
    std::string ptok = payloadToken;
    if( payloadToken.empty() ) ptok = "NA";
    rowData["PAYLOADTOKEN"].data< std::string >() = ptok;
    rowData["PROVENANCE"].data< std::string >() = userLogInfo.provenance;
    rowData["USERTEXT"].data< std::string >() = userLogInfo.usertext;
    rowData["IOVTAG"].data< std::string >() = iovtag;
    rowData["IOVTIMETYPE"].data< std::string >() = iovtimetype;
    rowData["PAYLOADINDEX"].data< unsigned int >() = payloadIdx;
    rowData["LASTSINCE"].data< unsigned long long >() = lastSince;
    rowData["EXECMESSAGE"].data< std::string >() = exceptionMessage;
    m_sessionHandle.nominalSchema().tableHandle(cond::LogDBNames::LogTableName()).dataEditor().insertRow(rowData);
  }catch(const std::exception& er){
    throw cond::Exception(std::string(er.what()));
  }
}

cond::Logger::~Logger(){
  if( m_sequenceManager ){
    delete m_sequenceManager;
    m_sequenceManager=0;
  }
}
