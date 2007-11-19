#include "CondCore/PopCon/interface/LogReader.h"

#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ITableDataEditor.h"

popcon::LogReader::LogReader (std::string pop_connect) : m_connect(pop_connect) {


	//connects to pop_con (metadata)schema 
	session=new cond::DBSession;
	session->configuration().setAuthenticationMethod( cond::XML );
	session->configuration().setMessageLevel( cond::Error );
	session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut(60);
	initialize();
}

popcon::LogReader::~LogReader ()
{	
	m_coraldb->disconnect();	
	delete m_coraldb;
	delete session;
}

void  popcon::LogReader::initialize()
{		
	try{
		session->open();
		m_coraldb = new cond::RelationalStorageManager(m_connect,session);
		m_coraldb->connect(cond::ReadOnly);
		m_coraldb->startTransaction(true);
	}catch(cond::Exception& er){
		std::cerr<< "LogReader::initialize cond " << er.what()<<std::endl;
		throw;
	}catch(std::exception& er){
		std::cerr<< "LogReader::initialize std " << er.what()<<std::endl;
		throw;
	}catch(...){
		std::cerr<<"Unknown error"<<std::endl;
	}
}

coral::TimeStamp  popcon::LogReader::lastRun(std::string& name, std::string& cs)
{
	coral::TimeStamp ts;
	try{
		coral::IQuery* query = m_coraldb->sessionProxy().nominalSchema().newQuery();
		query->addToOutputList("max(P_CON_EXECUTION.EXEC_START)","mes");
		query->addToTableList("P_CON_EXECUTION");
		query->addToTableList("P_CON_PAYLOAD_STATE");

		std::string condition = "P_CON_EXECUTION.OBJ_ID = P_CON_PAYLOAD_STATE.OBJ_ID AND P_CON_PAYLOAD_STATE.NAME =:nm AND P_CON_PAYLOAD_STATE.CONNECT_STRING =:cs";
		coral::AttributeList conditionData;
		conditionData.extend<std::string>( "nm" );
		conditionData.extend<std::string>( "cs" );
		query->setCondition( condition, conditionData );
		conditionData[0].data<std::string>() = name;
		conditionData[1].data<std::string>() = cs;
		
		query->setMemoryCacheSize( 100 );
		coral::ICursor& cursor = query->execute();
		if( cursor.next() ){
			const coral::AttributeList& row = cursor.currentRow();
			row.toOutputStream( std::cout ) << std::endl;
			ts =  row["mes"].data<coral::TimeStamp>();
			//std::cout << ts.day()  << " "<< ts.month() << " " << ts.year() << " " << ts.hour() << " " << ts.minute() << " " << ts.second() << " " << ts.nanosecond();
			//std::cout << " " << as << " " << id << std::endl;
		}
		cursor.close();
	}
	catch(std::exception& er){
		std::cerr << er.what();
	}
	return ts;
}
