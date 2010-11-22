#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/UserLogInfo.h"
#include "CondCore/DBCommon/interface/SequenceManager.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
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
bool
cond::Logger::getWriteLock()throw() {
  try{
    m_sessionHandle.transaction().start(false);
    coral::ITable& statusTable=m_sessionHandle.nominalSchema().tableHandle(LogDBNames::LogTableName());
    //Instructs the server to lock the rows involved in the result set.
    m_statusEditorHandle=statusTable.newQuery();
    m_statusEditorHandle->setForUpdate();
    m_statusEditorHandle->execute();
  }catch(const std::exception& er){
    delete m_statusEditorHandle;
    m_statusEditorHandle=0;
    return false;
  }
  m_locked=true;
  return true;
}
bool
cond::Logger::releaseWriteLock()throw() {
  if(m_locked){
    delete m_statusEditorHandle;
    m_statusEditorHandle=0;
  }
  m_locked=false;
  m_sessionHandle.transaction().commit();
  return !m_locked;
}
void 
cond::Logger::createLogDBIfNonExist(){
  if(m_logTableExists) return;
  m_sessionHandle.transaction().start(false);
  if(m_sessionHandle.nominalSchema().existsTable(cond::LogDBNames::SequenceTableName()) &&
     m_sessionHandle.nominalSchema().existsTable(cond::LogDBNames::LogTableName())){
    m_logTableExists=true;
    m_sessionHandle.transaction().commit();
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

  description.insertColumn(std::string("PAYLOADCONTAINER"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("PAYLOADCONTAINER"));

  description.insertColumn(std::string("PAYLOADNAME"),
	  coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("PAYLOADNAME"));

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
  m_sessionHandle.nominalSchema().createTable( description ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
  m_logTableExists=true;
  m_sessionHandle.transaction().commit();
}
void 
cond::Logger::logOperationNow(
			      const cond::UserLogInfo& userlogInfo,
			      const std::string& destDB,
			      const std::string& payloadToken,
			      const std::string& iovtag,
			      const std::string& iovtimetype,
			      unsigned int payloadIdx,
			      unsigned long long lastSince
			      ){
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
  this->insertLogRecord(targetLogId,now,destDB,payloadToken,userlogInfo,iovtag,iovtimetype,payloadIdx,lastSince,"OK");
}
void 
cond::Logger::logFailedOperationNow(
			       const cond::UserLogInfo& userlogInfo,
			       const std::string& destDB,
			       const std::string& payloadToken,
			       const std::string& iovtag,
			       const std::string& iovtimetype,
			       unsigned int payloadIdx,
			       unsigned long long lastSince,
			       const std::string& exceptionMessage
				    ){
  //aquirelocaltime
  boost::posix_time::ptime p=boost::posix_time::microsec_clock::local_time();
  std::string now=cond::to_string(p.date().year())+"-"+cond::to_string(p.date().month())+"-"+cond::to_string(p.date().day())+"-"+cond::to_string(p.time_of_day().hours())+":"+cond::to_string(p.time_of_day().minutes())+":"+cond::to_string(p.time_of_day().seconds());
  //aquireentryid
  if(!m_sequenceManager){
    m_sequenceManager=new cond::SequenceManager(m_sessionHandle,cond::LogDBNames::SequenceTableName());
  }
  unsigned long long targetLogId=m_sequenceManager->incrementId(LogDBNames::LogTableName());
  //insert log record with the new id
  this->insertLogRecord(targetLogId,now,destDB,payloadToken,userlogInfo,iovtag,iovtimetype,payloadIdx,lastSince,exceptionMessage);
}

void 
cond::Logger::LookupLastEntryByProvenance(const std::string& provenance,
					  LogDBEntry& logentry,
					  bool filterFailedOp) const{
  //select max(logid),etc from  logtable where etc"
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
    std::auto_ptr<coral::IQuery> query(m_sessionHandle.nominalSchema().newQuery());
    // construct select
    query->addToOutputList( cond::LogDBNames::LogTableName()+".LOGID" );
    query->defineOutputType( cond::LogDBNames::LogTableName()+".LOGID", "unsigned long long" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".DESTINATIONDB" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PROVENANCE" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".USERTEXT" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".IOVTAG" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".IOVTIMETYPE" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADINDEX" );
    query->defineOutputType( cond::LogDBNames::LogTableName()+".PAYLOADINDEX", "unsigned int" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".LASTSINCE" );
    query->defineOutputType( cond::LogDBNames::LogTableName()+".LASTSINCE", "unsigned long long" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADNAME" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADTOKEN" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADCONTAINER" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".EXECTIME" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".EXECMESSAGE" );
    
    coral::IQueryDefinition& subquery=query->defineSubQuery("subQ");
    query->addToTableList("subQ");
    query->addToTableList(cond::LogDBNames::LogTableName());
    subquery.addToTableList( cond::LogDBNames::LogTableName() );
    subquery.addToOutputList( "max(LOGID)", "max_logid");
    subquery.setCondition( whereClause, BindVariableList );
    query->setCondition(cond::LogDBNames::LogTableName()+std::string(".LOGID=subQ.max_logid"),coral::AttributeList() );
    query->defineOutputType( "subQ.max_logid", "unsigned long long" );
    
    coral::ICursor& cursor = query->execute();
    if( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      logentry.logId=row[cond::LogDBNames::LogTableName()+".LOGID"].data<unsigned long long>();
      logentry.destinationDB=row[cond::LogDBNames::LogTableName()+".DESTINATIONDB"].data<std::string>();
      logentry.provenance=row[cond::LogDBNames::LogTableName()+".PROVENANCE"].data<std::string>();
      logentry.usertext=row[cond::LogDBNames::LogTableName()+".USERTEXT"].data<std::string>();
      logentry.iovtag=row[cond::LogDBNames::LogTableName()+".IOVTAG"].data<std::string>();
      logentry.iovtimetype=row[cond::LogDBNames::LogTableName()+".IOVTIMETYPE"].data<std::string>();
      logentry.payloadIdx=row[cond::LogDBNames::LogTableName()+".PAYLOADINDEX"].data<unsigned int>();
      logentry.lastSince=row[cond::LogDBNames::LogTableName()+".LASTSINCE"].data<unsigned long long>();
      logentry.payloadName=row[cond::LogDBNames::LogTableName()+".PAYLOADNAME"].data<std::string>();
      logentry.payloadToken=row[cond::LogDBNames::LogTableName()+".PAYLOADTOKEN"].data<std::string>();
      logentry.payloadContainer=row[cond::LogDBNames::LogTableName()+".PAYLOADCONTAINER"].data<std::string>();
      logentry.exectime=row[cond::LogDBNames::LogTableName()+".EXECTIME"].data<std::string>();
      logentry.execmessage=row[cond::LogDBNames::LogTableName()+".EXECMESSAGE"].data<std::string>();
      
      //row.toOutputStream( std::cout ) << std::endl;
    }
  }
  m_sessionHandle.transaction().commit();
}
void 
cond::Logger::LookupLastEntryByTag( const std::string& iovtag,
				    const std::string & connectionStr,
				    cond::LogDBEntry& logentry,
				    bool filterFailedOp) const{
  /**
     select * from "COND_LOG_TABLE" where "LOGID"=(select max("LOGID") AS "max_logid" from "COND_LOG_TABLE" where "IOVTAG"='mytag1' and "EXECMESSAGE"='OK');
     
  */
  std::string whereClause=cond::LogDBNames::LogTableName();
  whereClause+=std::string(".IOVTAG=:iovtag");
  coral::AttributeList BindVariableList;
  BindVariableList.extend("iovtag",typeid(std::string) );
  BindVariableList["iovtag"].data<std::string>()=iovtag;
  if(connectionStr!=""){
    whereClause+=std::string(" AND ");
    whereClause+=cond::LogDBNames::LogTableName();
    whereClause+=std::string(".DESTINATIONDB=:destinationdb");
    BindVariableList.extend("destinationdb",typeid(std::string) ); 
    BindVariableList["destinationdb"].data<std::string>()=connectionStr;
  }
  if(filterFailedOp){
    whereClause+=std::string(" AND ");
    whereClause+=cond::LogDBNames::LogTableName();
    whereClause+=std::string(".EXECMESSAGE=:execmessage");
    BindVariableList.extend("execmessage",typeid(std::string) );
    BindVariableList["execmessage"].data<std::string>()="OK"; 
  }
  
  m_sessionHandle.transaction().start(true);
  //coral::IQuery* query = m_coraldb.nominalSchema().tableHandle(cond::LogDBNames::LogTableName()).newQuery();
  {
    std::auto_ptr<coral::IQuery> query(m_sessionHandle.nominalSchema().newQuery());
    // construct select
    query->addToOutputList( cond::LogDBNames::LogTableName()+".LOGID" );
    query->defineOutputType( cond::LogDBNames::LogTableName()+".LOGID", "unsigned long long" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".DESTINATIONDB" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PROVENANCE" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".USERTEXT" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".IOVTAG" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".IOVTIMETYPE" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADINDEX" );
    query->defineOutputType( cond::LogDBNames::LogTableName()+".PAYLOADINDEX", "unsigned int" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".LASTSINCE" );
    query->defineOutputType( cond::LogDBNames::LogTableName()+".LASTSINCE", "unsigned long long" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADNAME" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADTOKEN" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".PAYLOADCONTAINER" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".EXECTIME" );
    query->addToOutputList( cond::LogDBNames::LogTableName()+".EXECMESSAGE" );
    
    coral::IQueryDefinition& subquery=query->defineSubQuery("subQ");
    query->addToTableList("subQ");
    query->addToTableList(cond::LogDBNames::LogTableName());
    subquery.addToTableList( cond::LogDBNames::LogTableName() );
    subquery.addToOutputList( "max(LOGID)", "max_logid");
    subquery.setCondition( whereClause, BindVariableList );
    query->setCondition(cond::LogDBNames::LogTableName()+std::string(".LOGID=subQ.max_logid"),coral::AttributeList() );
    query->defineOutputType( "subQ.max_logid", "unsigned long long" );
    
    coral::ICursor& cursor = query->execute();
    if( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      logentry.logId=row[cond::LogDBNames::LogTableName()+".LOGID"].data<unsigned long long>();
      logentry.destinationDB=row[cond::LogDBNames::LogTableName()+".DESTINATIONDB"].data<std::string>();
      logentry.provenance=row[cond::LogDBNames::LogTableName()+".PROVENANCE"].data<std::string>();
      logentry.usertext=row[cond::LogDBNames::LogTableName()+".USERTEXT"].data<std::string>();
      logentry.iovtag=row[cond::LogDBNames::LogTableName()+".IOVTAG"].data<std::string>();
      logentry.iovtimetype=row[cond::LogDBNames::LogTableName()+".IOVTIMETYPE"].data<std::string>();
      logentry.payloadIdx=row[cond::LogDBNames::LogTableName()+".PAYLOADINDEX"].data<unsigned int>();
      logentry.lastSince=row[cond::LogDBNames::LogTableName()+".LASTSINCE"].data<unsigned long long>();
      logentry.payloadName=row[cond::LogDBNames::LogTableName()+".PAYLOADNAME"].data<std::string>();
      logentry.payloadToken=row[cond::LogDBNames::LogTableName()+".PAYLOADTOKEN"].data<std::string>();
      logentry.payloadContainer=row[cond::LogDBNames::LogTableName()+".PAYLOADCONTAINER"].data<std::string>();
      logentry.exectime=row[cond::LogDBNames::LogTableName()+".EXECTIME"].data<std::string>();
      logentry.execmessage=row[cond::LogDBNames::LogTableName()+".EXECMESSAGE"].data<std::string>();
      
      //row.toOutputStream( std::cout ) << std::endl;
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
			      const std::string& payloadToken,
			      const cond::UserLogInfo& userLogInfo,
			      const std::string& iovtag,
			      const std::string& iovtimetype,
			      unsigned int payloadIdx,
			      unsigned long long  lastSince,
			      const std::string& exceptionMessage){
  try{
    std::string containerName=parseToken(payloadToken).first;
    std::string payloadName=containerName; //now container and real class are assumed equal
    coral::AttributeList rowData;
    rowData.extend<unsigned long long>("LOGID");
    rowData.extend<std::string>("EXECTIME");
    rowData.extend<std::string>("PAYLOADCONTAINER");
    rowData.extend<std::string>("DESTINATIONDB");
    rowData.extend<std::string>("PAYLOADNAME");
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
    rowData["PAYLOADCONTAINER"].data< std::string >() = containerName;
    rowData["DESTINATIONDB"].data< std::string >() = destDB;
    rowData["PAYLOADNAME"].data< std::string >() = payloadName;
    rowData["PAYLOADTOKEN"].data< std::string >() = payloadToken;
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
