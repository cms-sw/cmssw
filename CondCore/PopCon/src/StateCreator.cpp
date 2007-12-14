#include "CondCore/PopCon/interface/StateCreator.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
//#include "CoralBase/Blob.h"
#include "SealBase/TimeInfo.h"
#include "CondCore/PopCon/interface/Exception.h"
#include <iostream>
#include <vector>
#include <stdexcept>

#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
popcon::StateCreator::StateCreator(const std::string& connectionString, 
				   const std::string& offlineString, 
				   const std::string& oname, 
				   bool dbg ):m_connect(connectionString),m_offline(offlineString), m_debug(dbg){
  if (m_debug)
    std::cerr<< "State creator: " << " Constructor\n";	
  nfo.object_name = oname;
  //do not check the state for sqlite DBs
  std::string::size_type loc = offlineString.find( "sqlite_", 0 );
  if( loc == std::string::npos ) {
    m_sqlite = false;
    conHandler.registerConnection(m_connect,m_connect,0);
    conHandler.registerConnection(m_offline,m_offline,0);
    session=new cond::DBSession;
    
    session->configuration().setAuthenticationMethod( cond::XML );
    if (m_debug)
      session->configuration().setMessageLevel( cond::Debug );
    else
      session->configuration().setMessageLevel( cond::Error );
    
    session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut(60);
    session->configuration().connectionConfiguration()->enableConnectionSharing();
    //session->connectionConfiguration().enableReadOnlySessionOnUpdateConnections();
    initialize();
  } else {
    m_sqlite = true;
  }
}

popcon::StateCreator::~StateCreator()
{ 
  if (!m_sqlite){
    //delete m_coraldb;
    delete session;
  }
}

void  popcon::StateCreator::initialize()
{		
  if (m_debug)
    std::cerr<< "State creator: " << " initialize\n";	
  if (m_sqlite)
    return;
  try{
    session->open();
  }catch(std::exception& er){
    //std::cerr<< " INITIALIZE EXCEPTION " <<er.what()<<std::endl;
    throw popcon::Exception(er.what());
  }catch(...){
    std::cerr<<"INITIALIZE Unknown error "<<std::endl;
    throw popcon::Exception("StateCreator::initialize unknown exception: ");
  }
}

void popcon::StateCreator::disconnect()
{
  //if (m_sqlite)
  //return;
}

bool popcon::StateCreator::compareStatusData()
{
  if (m_debug)
    std::cerr<< "State creator: " << " compare status data\n";	
  return (m_current_state == m_saved_state);	
}

void popcon::StateCreator::storeStatusData()
{
  if (m_debug)
    std::cerr<< "State creator: " << " store status data\n";	
  if (m_sqlite)
    return;
  cond::CoralTransaction& coraldb=conHandler.getConnection(m_connect)->coralTransaction();
  try{	
    coraldb.start(true);
    coral::ITable& mytable=coraldb.coralSessionProxy().nominalSchema().tableHandle("P_CON_PAYLOAD_STATE");
    coral::ITableDataEditor& editor = mytable.dataEditor();
    
    coral::AttributeList rowBuffer;
    std::string ua;
    std::string uc;
    
    //return rowbuffer from object
    rowBuffer = m_current_state.update_helper(ua,uc);
    
    int rowsUpdated = editor.updateRows( ua, uc, rowBuffer );
    if ( rowsUpdated != 1 ) {
      throw std::runtime_error( "Unexpected number of rows updated" );
    }
    coraldb.commit();
  }catch(coral::Exception& er){
    //std::cerr <<"StateCreator::storeStatusData Coral exception: " << er.what();
    coraldb.rollback();
    throw popcon::Exception("caught Coral exception in StateCreator::generateStatusData exception: "+ (std::string)er.what());
  }
}

void popcon::StateCreator::generateStatusData()
{
  if (m_debug)
    std::cerr<< "State creator: " << " generate status data\n";	
  if (m_sqlite)
    return;
  std::cout<<"StateCreator::generateStatusData"<<std::endl;

  if (nfo.top_level_table == "")
    getPoolTableName();

  if (nfo.top_level_table == "") {
    m_current_state = DBState(nfo.object_name, 0, m_offline);
    return;
  }
  
  cond::CoralTransaction& status_db=conHandler.getConnection(m_connect)->coralTransaction();
  try{
    status_db.start(true);    
    coral::AttributeList rowBuffer;
    coral::ICursor* cursor; 
    rowBuffer.extend<int>( "S" );
    coral::ISchema& schema = status_db.coralSessionProxy().nominalSchema();
    //std::auto_ptr< coral::IQuery > query(schema.newQuery()); 
    coral::IQuery* query(schema.newQuery()); 
    query->addToOutputList( "count(*)","S");
    query->addToTableList(nfo.top_level_table,"TLT");
    query->setMemoryCacheSize( 5 );
    query->defineOutput( rowBuffer );
    cursor = &(query->execute());
    if (cursor->next() ){
      if (m_debug)
	std::cerr << "generateStatusData()for " << nfo.object_name << " , size is:  " <<  rowBuffer[0].data<int>() << std::endl;
      m_current_state = DBState(nfo.object_name, rowBuffer[0].data<int>(), m_offline);
    }	
    status_db.commit();
    //status_db.disconnect();
    //delete query;
  }catch(popcon::Exception& e){
    throw popcon::Exception("caught previously thrown popcon exception in StateCreator::generateStatusData: "+(std::string)e.what());
  }catch(coral::Exception& er){
    status_db.rollback();
    //std::cerr <<"StateCreator::generateStatusData Coral exception: " << er.what();
    throw popcon::Exception("caught Coral exception in StateCreator::generateStatusData: "+(std::string)er.what());
  }
}

void popcon::StateCreator::getPoolTableName()
{
  if (m_debug)
    std::cerr<< "State creator: " << " get pool data\n";	
  cond::CoralTransaction& offline_db=conHandler.getConnection(m_offline)->coralTransaction();
  try{
    offline_db.start(true);
    coral::ISchema& schema = offline_db.coralSessionProxy().nominalSchema();

    if ( ! schema.existsTable("POOL_RSS_CONTAINERS")){
      nfo.top_level_table = "";
      offline_db.commit();
      return;
    }

    coral::AttributeList rowBuffer;
    rowBuffer.extend<std::string>( "TLT" );
    //std::auto_ptr< coral::IQuery > query(schema.newQuery()); 
    coral::IQuery* query(schema.newQuery()); 
    query->addToOutputList( "P.TABLE_NAME","TLT");
    query->addToTableList("POOL_RSS_CONTAINERS","P");
    coral::AttributeList conditionData;
    conditionData.extend<std::string>( "oname" );
    conditionData[0].data<std::string>() = nfo.object_name;
    std::string condition = "P.CLASS_NAME = :oname";
    query->setCondition(condition, conditionData);
    query->setMemoryCacheSize( 5 );
    query->defineOutput( rowBuffer );
    coral::ICursor& cursor4 = query->execute();
    if (cursor4.next()){
      nfo.top_level_table = rowBuffer[0].data<std::string>();
      if (m_debug)
	std::cerr << "Top level table is " << nfo.top_level_table << std::endl;
    }
    offline_db.commit();
    //status_db.disconnect();
    //delete query;
  }catch(coral::Exception& er){
    offline_db.rollback();
    //std::cerr << "StateCreator::getPoolTableName Coral exception: " << er.what();
    throw popcon::Exception("caught Coral exception in StateCreator::getPoolTableName : "+(std::string)er.what());
  }
}

bool popcon::StateCreator::previousExceptions(bool& fix){
  if (m_debug)
    std::cerr<< "State creator: " << " previous exceptions\n";	
  if (m_sqlite)
    return false;
  
  getStoredStatusData();
  if(m_saved_state.except_description != "")	//exception
    {
      if (m_saved_state.manual_override  != "")
	fix = true;
      return true;
    }
  return false;
}


bool popcon::StateCreator::checkAndCompareState()
{
  if (m_sqlite)
    return true;
  getPoolTableName();
  generateStatusData();
  return compareStatusData();
}

void  popcon::StateCreator::setException(std::string ex)
{
  if (m_debug)
    std::cerr<< "State creator: " << " check and compare state\n";	
  m_current_state.except_description = ex;
}


void popcon::StateCreator::getStoredStatusData()
{
  if (m_debug)
    std::cerr<< "State creator: " << "getStoredStatusData\n";	
  if (m_sqlite)
    return;
  cond::CoralTransaction& coraldb=conHandler.getConnection(m_connect)->coralTransaction();
  try{
    coraldb.start(true);
    coral::ISchema& schema = coraldb.coralSessionProxy().nominalSchema();
    coral::AttributeList rowBuffer;
    rowBuffer.extend<std::string>( "N" );
    rowBuffer.extend<int>( "PS" );
    rowBuffer.extend<std::string>( "ED" );
    rowBuffer.extend<std::string>( "MO" );
    coral::IQuery* query = schema.newQuery();
    query->addToOutputList( "P_CON_PAYLOAD_STATE.NAME","N");
    query->addToOutputList( "P_CON_PAYLOAD_STATE.PAYLOAD_SIZE","PS");
    query->addToOutputList( "P_CON_PAYLOAD_STATE.EXCEPT_DESCRIPTION","ED");
    query->addToOutputList( "P_CON_PAYLOAD_STATE.MANUAL_OVERRIDE","MO");
    query->addToTableList("P_CON_PAYLOAD_STATE","cps");
    coral::AttributeList conditionData;
    conditionData.extend<std::string>( "oname" );
    conditionData[ "oname" ].data<std::string>() = nfo.object_name;
    conditionData.extend<std::string>( "connStr" );
    conditionData[ "connStr" ].data<std::string>() = m_offline;
    std::string condition = "cps.NAME = :oname and cps.CONNECT_STRING = :connStr";
    
    query->setCondition(condition, conditionData);
    query->setMemoryCacheSize( 100 );
    query->defineOutput( rowBuffer );
    coral::ICursor& cursor4 = query->execute();
    unsigned short count=0;
    while ( cursor4.next() ) {
      m_saved_state=DBState(rowBuffer, m_offline);
      count ++;
    }
    coraldb.commit();	
    delete query;
	if (count != 1){
	  throw popcon::Exception("cannot find ObjectName in the database");
	}
  }catch(coral::Exception& er){
    coraldb.rollback();;
    //std::cerr << "StateCreator::getStoredStatusData Coral exception: " << er.what() << std::endl;
    throw popcon::Exception("caught Coral exception in StateCreator::getStoredStatusData:  "+(std::string)er.what());
  }
}
