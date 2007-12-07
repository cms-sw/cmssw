#include "CondCore/PopCon/interface/Logger.h"

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
#include <string>


popcon::Logger::Logger (std::string connectionString, std::string offlineString,std::string name, bool dbg) : m_obj_name(name), m_connect(connectionString), 
m_offline(offlineString), m_debug(dbg), m_established(false)  {

	//W A R N I N G - session has to be alive throughout object lifetime
	//otherwise there will be problems with currvals of the sequences

	std::string::size_type loc = m_offline.find( "sqlite_file", 0 );
	if( loc == std::string::npos ) {
		m_sqlite = false;
		session=new cond::DBSession(false);
		session->sessionConfiguration().setAuthenticationMethod( cond::XML );
		if (m_debug)
			session->sessionConfiguration().setMessageLevel( cond::Debug );
		else	
			session->sessionConfiguration().setMessageLevel( cond::Error );
		session->connectionConfiguration().setConnectionRetrialTimeOut(60);
		session->connectionConfiguration().enableConnectionSharing();
		session->connectionConfiguration().enableReadOnlySessionOnUpdateConnections();
		initialize();
	}
	else{ 
		m_sqlite=true;
	}
}

popcon::Logger::~Logger ()
{	
	if (!m_sqlite)
	{
		disconnect();
		delete m_coraldb;
		delete session;
	}
}

void  popcon::Logger::initialize()
{		
	try{

		if (m_debug)
			std::cerr << "Logger::initialize - session.open\n";
		session->open();
		m_coraldb = new cond::RelationalStorageManager(m_connect,session);
		m_coraldb->connect(cond::ReadWrite);
		m_coraldb->startTransaction(false);
		m_established = true;
		//FIXME - subquery instead
		payloadIDMap();

	}catch(cond::Exception& er){
		std::cerr<< "Logger::initialize cond " << er.what()<<std::endl;
	}catch(std::exception& er){
		std::cerr<< "Logger::initialize std " << er.what()<<std::endl;
	}catch(...){
		std::cerr<<"Unknown error"<<std::endl;
	}
}



void popcon::Logger::disconnect()
{
	if (m_sqlite)
		return;
	if (!m_established)
	{
		std::cerr << " Logger::disconnect - connection has not been established, skipping\n";
		return;
	}
	if (m_debug)
		std::cerr << "Disconnecting\n";
	m_coraldb->disconnect();	
	if (m_debug)
		std::cerr << "Disconnected\n";
}

void popcon::Logger::payloadIDMap()
{
	if (m_debug)
		std::cerr << "PayloadIDMap\n";
	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_PAYLOAD_STATE");
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
	catch(std::exception& er){
		std::cerr << er.what();
	}
}

void popcon::Logger::lock()
{
	if (m_sqlite)
		return;
	std::cerr<< " Locking\n";
	if (!m_established)
		throw popcon::Exception("Logger::lock exception ");
	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_LOCK");
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
	catch(std::exception& er){
		std::cerr << "Logger::lock " << er.what();
		throw popcon::Exception("Logger::lock exception ");
	}
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

	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION");
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
	catch(std::exception& er){
		std::cerr << er.what();
	}
}

void popcon::Logger::updatePayloadID()
{
	if (m_debug)
		std::cerr << "Logger::updatePayloadID\n";
	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION_PAYLOAD");
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
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
void popcon::Logger::newExecution()
{
	if (m_sqlite)
		return;
	if (m_debug)
		std::cerr << "Logger::newExecution\n";

	if (!m_established)
		throw popcon::Exception("Logger::newExecution log exception ");
	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION");
		coral::AttributeList rowBuffer;
		coral::ITableDataEditor& dataEditor = mytable.dataEditor();
		dataEditor.rowBuffer( rowBuffer );
		rowBuffer["OBJ_ID"].data<int>()=m_id_map[m_obj_name];
		rowBuffer["EXEC_ID"].data<int>()= -1;
		rowBuffer["EXEC_START"].data<coral::TimeStamp>() = coral::TimeStamp::now();
		dataEditor.insertRow( rowBuffer );
	}
	catch(coral::Exception& er)
	{
		std::cerr << " Probably there's no entry related to " << m_obj_name << " " << er.what() << std::endl;
		throw popcon::Exception("Logger::newExecution log exception ");
	}
	catch(std::exception& er){
		std::cerr << er.what();
		throw popcon::Exception("Logger::newExecution log exception ");
	}
}
void popcon::Logger::newPayload()
{
	if (m_sqlite)
		return;
	if (m_debug)
		std::cerr << "Logger::newPayload\n";
	coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION_PAYLOAD");
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
	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION");
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
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
void popcon::Logger::finalizePayload(std::string ok)
{
	if (m_sqlite)
		return;
	if (m_debug)
		std::cerr << "Logger::finalizePayload\n";
	updatePayloadID();
	try{
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("P_CON_EXECUTION_PAYLOAD");
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
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
