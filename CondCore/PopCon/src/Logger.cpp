#include "CondCore/PopCon/interface/Logger.h"
#include <memory>
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/Exception.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/TimeStamp.h"

#include "CondCore/PopCon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Exception.h"
static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
popcon::Logger::Logger (const std::string& connectionString, 
			const std::string& offlineString,
			const std::string& payloadName, 
			bool dbg) : m_obj_name(payloadName), m_connect(connectionString), m_debug(dbg), m_established(false)  {

  //W A R N I N G - session has to be alive throughout object lifetime
  //otherwise there will be problems with currvals of the sequences
  std::string::size_type loc = offlineString.find( "sqlite_", 0 );
  if( loc == std::string::npos ) {
    m_sqlite = false;
    conHandler.registerConnection(m_connect,m_connect,0);
    session=new cond::DBSession;
    session->configuration().setAuthenticationMethod( cond::XML );
    if (m_debug){
      session->configuration().setMessageLevel( cond::Debug );
    }else{	
      session->configuration().setMessageLevel( cond::Error );
    }
    session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut(60);
    session->configuration().connectionConfiguration()->enableConnectionSharing();
    session->configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections();
    session->open();
    conHandler.connect(session);
    m_coraldb=&(conHandler.getConnection(m_connect)->coralTransaction());
    m_established = true;
    //FIXME - subquery instead
    m_coraldb->start(true);
    payloadIDMap();
    m_coraldb->commit();
  }else{ 
    m_sqlite=true;
  }
}

popcon::Logger::~Logger ()
{	
  if (!m_sqlite)
    {
      disconnect();
      delete session;
    }
}

void  popcon::Logger::initialize()
{		
  //  if (m_debug) std::cerr << "Logger::initialize - session.open\n";
}

void popcon::Logger::disconnect()
{
  if (m_sqlite) return;
  if (!m_established){
    std::cerr << " Logger::disconnect - connection has not been established, skipping\n";
    return;
  }
  if (m_debug) std::cerr << "Disconnected\n";
  //m_coraldb->commit();
}

void popcon::Logger::payloadIDMap()
{
  if (m_debug){
    std::cerr << "PayloadIDMap\n";
  }
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_PAYLOAD_STATE");
  std::auto_ptr< coral::IQuery > query(mytable.newQuery());
  query->addToOutputList("NAME");
  query->addToOutputList("OBJ_ID");
  query->setMemoryCacheSize( 100 );
  coral::ICursor& cursor = query->execute();
  while( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    m_id_map.insert(std::make_pair(row["NAME"].data<std::string>(), row["OBJ_ID"].data<int>()));
  }
  cursor.close();
}

void popcon::Logger::lock()
{
  if (m_sqlite)  return;
  std::cerr<< " Locking\n";
  if (!m_established){
    throw popcon::Exception("Logger::lock exception ");
  }
  m_coraldb->start(false);
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_LOCK");
  coral::AttributeList inputData;
  coral::ITableDataEditor& dataEditor = mytable.dataEditor();
  inputData.extend<int>("id");
  inputData.extend<std::string>("name");
  inputData[0].data<int>() = 69;
  inputData[1].data<std::string>() = m_obj_name;
  std::string setClause("LOCK_ID = :id");
  std::string condition("OBJECT_NAME = :name");
  dataEditor.updateRows( setClause, condition, inputData );
  //DO NOT COMMIT - DBMS holds the row exclusive lock till commit
}

void popcon::Logger::unlock()
{
  if (m_sqlite)
    return;
  if (!m_established)
    return;
  std::cerr<< " Unlocking\n";
  m_coraldb->commit();	
}

void popcon::Logger::updateExecID()
{
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION");
  std::auto_ptr< coral::IQuery > query(mytable.newQuery());
  query->addToOutputList("max(EXEC_ID)");
  query->setMemoryCacheSize( 100 );
  coral::ICursor& cursor = query->execute();
  while( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    m_exec_id = (int) row[0].data<double>();
  }
  cursor.close();
}

void popcon::Logger::updatePayloadID()
{
  if (m_debug){
    std::cerr << "Logger::updatePayloadID\n"; 
  }
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION_PAYLOAD");
  std::auto_ptr< coral::IQuery > query(mytable.newQuery());
  query->addToOutputList("max(PL_ID)");
  query->setMemoryCacheSize( 100 );
  coral::ICursor& cursor = query->execute();
  while( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    m_payload_id = (int) row[0].data<double>();
  }
  cursor.close();
}
void popcon::Logger::newExecution()
{
  if (m_sqlite) return;
  if (m_debug) std::cerr << "Logger::newExecution\n";
  if (!m_established)
    throw popcon::Exception("Logger::newExecution log exception ");
  
  //m_coraldb->start();
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION");
  coral::AttributeList rowBuffer;
  coral::ITableDataEditor& dataEditor = mytable.dataEditor();
  dataEditor.rowBuffer( rowBuffer );
  rowBuffer["OBJ_ID"].data<int>()=m_id_map[m_obj_name];
  rowBuffer["EXEC_ID"].data<int>()= -1;
  rowBuffer["EXEC_START"].data<coral::TimeStamp>() = coral::TimeStamp::now();
  dataEditor.insertRow( rowBuffer );
}
void popcon::Logger::newPayload()
{
  if (m_sqlite)
    return;
  if (m_debug)
    std::cerr << "Logger::newPayload\n";
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION_PAYLOAD");
  coral::AttributeList rowBuffer;
  coral::ITableDataEditor& dataEditor = mytable.dataEditor();
  dataEditor.rowBuffer( rowBuffer );
  rowBuffer["EXCEPT_DESCRIPTION"].data<std::string>()= "Fetched but not written";
  rowBuffer["PL_ID"].data<int>()= -1; 
  rowBuffer["EXEC_ID"].data<int>()= -1;
  dataEditor.insertRow( rowBuffer );
}

void popcon::Logger::finalizeExecution(std::string ok)
{
  if (m_sqlite)
    return;
  if (m_debug)
    std::cerr << "Logger::finalizeExecution\n";
  if (!m_established)
    {
      std::cerr << " Logger::finalizeExecution - connection has not been established, skipping\n";
      return;
    }
  updateExecID();
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION");
  coral::AttributeList inputData;
  coral::ITableDataEditor& dataEditor = mytable.dataEditor();
  inputData.extend<coral::TimeStamp>("newEnd");
  inputData.extend<std::string>("newStatus");
  inputData.extend<int>("last_id");
  inputData[0].data<coral::TimeStamp>() = coral::TimeStamp::now();
  inputData[1].data<std::string>() = ok;
  inputData[2].data<int>() = m_exec_id;
  std::string setClause("EXEC_END = :newEnd ,status = :newStatus" );
  std::string condition("EXEC_ID = :last_id" );
  dataEditor.updateRows( setClause, condition, inputData );
}
void popcon::Logger::finalizePayload(std::string ok)
{
  if (m_sqlite)
    return;
  if (m_debug)
    std::cerr << "Logger::finalizePayload\n";
  updatePayloadID();
  coral::ITable& mytable=m_coraldb->coralSessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION_PAYLOAD");
  coral::AttributeList inputData;
  coral::ITableDataEditor& dataEditor = mytable.dataEditor();
  inputData.extend<coral::TimeStamp>("newEnd");
  inputData.extend<std::string>("newStatus");
  inputData.extend<int>("last_id");
  inputData[0].data<coral::TimeStamp>() = coral::TimeStamp::now();
  if (ok != "OK")
    inputData[0].setNull();
  inputData[1].data<std::string>() = ok;
  inputData[2].data<int>() = m_payload_id;
  std::string setClause("WRITTEN = :newEnd, EXCEPT_DESCRIPTION = :newStatus" );
  std::string condition("PL_ID = :last_id" );
  dataEditor.updateRows( setClause, condition, inputData );
}
