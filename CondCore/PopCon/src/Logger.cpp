#include "CondCore/PopCon/interface/Logger.h"

#include "RelationalAccess/ISessionProxy.h"
//#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
//#include "RelationalAccess/IPrimaryKey.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ITableDataEditor.h"
//#include "CoralBase/Exception.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/TimeStamp.h"

#include "CondCore/PopCon/interface/Exception.h"


popcon::Logger::Logger (std::string connectionString,std::string name, bool dbg) : m_obj_name(name), m_connect(connectionString), m_debug(dbg) {

	//W A R N I N G - session has to be alive throughout object lifetime
	//otherwiese there will be problems with currvals of the sequences
	

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
	//FIXME - subquery instead
	payloadIDMap();
}

popcon::Logger::~Logger ()
{	
	delete m_coraldb;
	delete session;
}

void  popcon::Logger::initialize()
{		
	try{

		session->open();
		m_coraldb = new cond::RelationalStorageManager(m_connect,session);
		m_coraldb->connect(cond::ReadWrite);
		m_coraldb->startTransaction(false);

	}catch(cond::Exception& er){
		std::cerr<< "Logger::initialize " << er.what()<<std::endl;
		throw popcon::Exception("Logger::initialize cond::Exception ");
	}catch(std::exception& er){
		std::cerr<< "Logger::initialize " << er.what()<<std::endl;
		throw popcon::Exception("Logger::initialize std::exception ");
	}catch(...){
		std::cerr<<"Unknown error"<<std::endl;
		throw popcon::Exception("Logger::initialize unknown error ");
	}
}



void popcon::Logger::disconnect()
{
	if (m_debug)
		std::cerr << "Disconnecting\n";
	m_coraldb->disconnect();	
	if (m_debug)
		std::cerr << "Disconnected\n";
}

void popcon::Logger::payloadIDMap()
{
	//FIXME uneffective, insert with join or subquery instead
	//fill m_id_map
	if (m_debug)
		std::cerr << "PayloadIDMap\n";
	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_PAYLOAD_STATE");
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
		//m_coraldb->commit();	
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
}

void popcon::Logger::lock()
{
	std::cerr<< " Locking\n";
	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_LOCK");
		coral::AttributeList inputData;
		coral::ITableDataEditor& dataEditor = mytable.dataEditor();
		inputData.extend<int>("id");
		inputData.extend<std::string>("name");
		inputData[0].data<int>() = 69;
		inputData[1].data<std::string>() = m_obj_name;
		std::string setClause("LOCK_COL = :id");
		std::string condition("NAME = :name");
		dataEditor.updateRows( setClause, condition, inputData );
		//DO NOT COMMIT
		////m_coraldb->commit();	
	}
	catch(std::exception& er){
		std::cerr << "Logger::lock " << er.what();
	}
}

void popcon::Logger::unlock()
{
	
	std::cerr<< " Unlocking\n";
	m_coraldb->commit();	
}

void popcon::Logger::blah(std::string s)
{
	std::cerr << "Blah of " << s << std::endl;
}


void popcon::Logger::updateExecID()
{

	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_EXECUTION");
		std::auto_ptr< coral::IQuery > query(mytable.newQuery());
		query->addToOutputList("max(EXEC_ID)");
		query->setMemoryCacheSize( 100 );
		coral::ICursor& cursor = query->execute();
		while( cursor.next() ){
			const coral::AttributeList& row = cursor.currentRow();
			m_exec_id = (int) row[0].data<double>();
		}
		cursor.close();
		//m_coraldb->commit();
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
}

void popcon::Logger::updatePayloadID()
{
	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_EXECUTION_PAYLOAD");
		std::auto_ptr< coral::IQuery > query(mytable.newQuery());
		query->addToOutputList("max(PL_ID)");
		query->setMemoryCacheSize( 100 );
		coral::ICursor& cursor = query->execute();
		while( cursor.next() ){
			const coral::AttributeList& row = cursor.currentRow();
			m_payload_id = (int) row[0].data<double>();
		}
		cursor.close();
		//m_coraldb->commit();
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
void popcon::Logger::newExecution()
{
	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_EXECUTION");
		coral::AttributeList rowBuffer;
		coral::ITableDataEditor& dataEditor = mytable.dataEditor();
		dataEditor.rowBuffer( rowBuffer );
		rowBuffer["OBJ_ID"].data<int>()=m_id_map[m_obj_name];
		rowBuffer["EXEC_ID"].data<int>()= -1;
		rowBuffer["EXEC_START"].data<coral::TimeStamp>() = coral::TimeStamp::now();
		dataEditor.insertRow( rowBuffer );
		//m_coraldb->commit();	
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
void popcon::Logger::newPayload()
{
	//m_coraldb->startTransaction(false);
	coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_EXECUTION_PAYLOAD");
	coral::AttributeList rowBuffer;
	coral::ITableDataEditor& dataEditor = mytable.dataEditor();
	dataEditor.rowBuffer( rowBuffer );
	rowBuffer["EXCEPT_DESCRIPTION"].data<std::string>()= "Fetched but not written";
	rowBuffer["PL_ID"].data<int>()= -1; 
	rowBuffer["EXEC_ID"].data<int>()= -1;
	dataEditor.insertRow( rowBuffer );

	//m_coraldb->commit();	

}


void popcon::Logger::finalizeExecution(std::string ok)
{
	updateExecID();
	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_EXECUTION");
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
		//m_coraldb->commit();	
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
void popcon::Logger::finalizePayload(std::string ok)
{
	updatePayloadID();
	try{
		//m_coraldb->startTransaction(false);
		coral::ITable& mytable=m_coraldb->sessionProxy().nominalSchema().tableHandle("O2O_EXECUTION_PAYLOAD");
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
		//m_coraldb->commit();	
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
}
