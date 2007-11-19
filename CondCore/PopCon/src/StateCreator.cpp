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
#include "CoralBase/Blob.h"
#include "SealBase/TimeInfo.h"
#include "CondCore/PopCon/interface/Exception.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>


popcon::StateCreator::StateCreator(std::string connectionString, std::string offlineString, std::string oname, bool dbg ):m_connect(connectionString), m_offline(offlineString), m_debug(dbg)
{
	if (m_debug)
		std::cerr<< "State creator: " << " Constructor\n";	

	nfo.object_name = oname;
	
	//do not check the state for sqlite DBs
	std::string::size_type loc = m_offline.find( "sqlite_file", 0 );
	if( loc == std::string::npos ) {
		m_sqlite = false;

		session=new cond::DBSession();

		session->configuration().setAuthenticationMethod( cond::XML );
		if (m_debug)
		  session->configuration().setMessageLevel( cond::Debug );
		else
		  session->configuration().setMessageLevel( cond::Error );
		
		session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut(60);
		session->configuration().connectionConfiguration()->enableConnectionSharing();
		//session->connectionConfiguration().enableReadOnlySessionOnUpdateConnections();
		initialize();
	}
	else 
		m_sqlite = true;
}

popcon::StateCreator::~StateCreator()
{ 
	if (!m_sqlite){
		delete m_coraldb;
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
		m_coraldb = new cond::RelationalStorageManager(m_connect,session);
		m_coraldb->connect(cond::ReadWrite);

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
	if (m_sqlite)
		return;
	m_coraldb->disconnect();	
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
	try{	
		m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_PAYLOAD_STATE");
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
		m_coraldb->commit();
	}	
	catch(coral::Exception& er){
		//std::cerr <<"StateCreator::storeStatusData Coral exception: " << er.what();
		this->disconnect();
		throw popcon::Exception("caught Coral exception in StateCreator::generateStatusData exception: "+ (std::string)er.what());
	}
}

void popcon::StateCreator::generateStatusData()
{
	if (m_debug)
		std::cerr<< "State creator: " << " generate status data\n";	
	if (m_sqlite)
		return;

	try{
		cond::RelationalStorageManager status_db(m_offline, /*cond*/session);
		status_db.connect(cond::ReadOnly);

		if (nfo.top_level_table == "")
			getPoolTableName();

		status_db.startTransaction();

		coral::AttributeList rowBuffer;
		coral::ICursor* cursor; 
		rowBuffer.extend<int>( "S" );
		coral::ISchema& schema = status_db.sessionProxy().nominalSchema();
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

		//status_db.commit();
		status_db.disconnect();
		//delete query;
	}
	catch(popcon::Exception& e)
	{
		throw popcon::Exception("caught previously thrown popcon exception in StateCreator::generateStatusData: "+(std::string)e.what());
	}
	catch(coral::Exception& er){
		this->disconnect();
		//std::cerr <<"StateCreator::generateStatusData Coral exception: " << er.what();
		throw popcon::Exception("caught Coral exception in StateCreator::generateStatusData: "+(std::string)er.what());
	}


}

void popcon::StateCreator::getPoolTableName()
{
	if (m_debug)
		std::cerr<< "State creator: " << " get pool data\n";	
	try{
		cond::RelationalStorageManager status_db(m_offline,/*cond*/session);
		status_db.connect(cond::ReadOnly);
		status_db.startTransaction();

		coral::AttributeList rowBuffer;
		rowBuffer.extend<std::string>( "TLT" );
		coral::ISchema& schema = status_db.sessionProxy().nominalSchema();
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
		//status_db.commit();
		status_db.disconnect();
		//delete query;
	}
	catch(coral::Exception& er){
		this->disconnect();
		//std::cerr << "StateCreator::getPoolTableName Coral exception: " << er.what();
		throw popcon::Exception("caught Coral exception in StateCreator::getPoolTableName : "+(std::string)er.what());
	}
}

bool popcon::StateCreator::previousExceptions(bool& fix)
{
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
	try{
		m_coraldb->startTransaction(false);
		coral::ISchema& schema = m_coraldb->sessionProxy().nominalSchema();
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
		query->addToTableList("P_CON_PAYLOAD_STATE","");
		coral::AttributeList conditionData;
		conditionData.extend<std::string>( "oname" );
		conditionData[0].data<std::string>() = nfo.object_name;
		std::string condition = "P_CON_PAYLOAD_STATE.NAME = :oname";
		query->setCondition(condition, conditionData);
		query->setMemoryCacheSize( 100 );
		query->defineOutput( rowBuffer );
		coral::ICursor& cursor4 = query->execute();
		unsigned short count=0;
		while ( cursor4.next() ) {
			m_saved_state=DBState(rowBuffer, m_offline);
			count ++;
		}

		m_coraldb->commit();	
		delete query;

		if (count != 1){
			throw std::runtime_error("cannot find ObjectName in the database");
		}
	}
	catch(coral::Exception& er){
		this->disconnect();
		//std::cerr << "StateCreator::getStoredStatusData Coral exception: " << er.what() << std::endl;
		throw popcon::Exception("caught Coral exception in StateCreator::getStoredStatusData:  "+(std::string)er.what());
	}
}
