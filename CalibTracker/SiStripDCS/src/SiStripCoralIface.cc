#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"

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

SiStripCoralIface::SiStripCoralIface (std::string connectionString , std::string authenticationPath) : m_connect(connectionString) {

	session=new cond::DBSession(false);
	session->sessionConfiguration().setAuthenticationMethod( cond::XML );
        session->sessionConfiguration().setAuthenticationPath(authenticationPath);
	session->sessionConfiguration().setMessageLevel( cond::Error );
	session->connectionConfiguration().setConnectionRetrialTimeOut(60);
	initialize();
}

SiStripCoralIface::~SiStripCoralIface ()
{	
	m_coraldb->disconnect();	
	delete m_coraldb;
	delete session;
}

void  SiStripCoralIface::initialize()
{		
		session->open();
		m_coraldb = new cond::RelationalStorageManager(m_connect,session);
		m_coraldb->connect(cond::ReadOnly);
		m_coraldb->startTransaction(true);

}

void SiStripCoralIface::doQuery(coral::TimeStamp startTime, coral::TimeStamp endTime, std::vector<coral::TimeStamp> &vec_changedate, std::vector<uint32_t>  &vec_dpid, std::vector<uint32_t>  &vec_actualStatus)
{
                coral::IQuery* query = m_coraldb->sessionProxy().nominalSchema().newQuery();
                query->addToOutputList("FWCAENCHANNEL.CHANGE_DATE","CHANGE_DATE");
                query->addToOutputList("FWCAENCHANNEL.ACTUAL_STATUS","ACTUAL_STATUS");
                query->addToOutputList("FWCAENCHANNEL.DPID","DPID");
                query->addToOrderList("FWCAENCHANNEL.CHANGE_DATE");
                query->addToTableList("FWCAENCHANNEL");
                query->addToTableList("DP_NAME2ID");
		std::string condition = "FWCAENCHANNEL.DPID = DP_NAME2ID.id AND FWCAENCHANNEL.CHANGE_DATE<=:tmax AND FWCAENCHANNEL.ACTUAL_STATUS IS NOT NULL AND FWCAENCHANNEL.CHANGE_DATE >=:tmin AND (DP_NAME2ID.dpname like '%channel002%' or DP_NAME2ID.dpname like '%channel003%')";

                coral::AttributeList conditionData;
                conditionData.extend<coral::TimeStamp>( "tmax" );
                conditionData.extend<coral::TimeStamp>( "tmin" );

                query->setCondition( condition, conditionData );
                conditionData[0].data<coral::TimeStamp>() = endTime;
                conditionData[1].data<coral::TimeStamp>() = startTime;

		query->setMemoryCacheSize( 100 );
		coral::ICursor& cursor = query->execute();
		int numberRow=0;
		while( cursor.next() ){
			const coral::AttributeList& row = cursor.currentRow();
			
			row.toOutputStream( std::cout ) << std::endl;
			numberRow++;
			coral::TimeStamp ts =  row["CHANGE_DATE"].data<coral::TimeStamp>();
			vec_changedate.push_back(ts);
			uint32_t as = (uint32_t)row["ACTUAL_STATUS"].data<float>();
			vec_actualStatus.push_back(as);
			uint32_t id = (uint32_t)row["DPID"].data<float>();
			vec_dpid.push_back(id);
			//std::cout << ts.day()  << " "<< ts.month() << " " << ts.year() << " " << ts.hour() << " " << ts.minute() << " " << ts.second() << " " << ts.nanosecond();
			//std::cout << " " << as << " " << id << std::endl;
		}
		cursor.close();
		std::cout<<"This query returns "<<numberRow<<" rows"<<std::endl;
}
