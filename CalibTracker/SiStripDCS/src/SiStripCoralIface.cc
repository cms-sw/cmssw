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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// constructor
SiStripCoralIface::SiStripCoralIface( std::string connectionString , std::string authenticationPath, const bool debug) : m_connectionString(connectionString), m_session(), debug_(debug)
{
  std::cout << "Building coral interface" << std::endl;
  m_connection.configuration().setAuthenticationPath(authenticationPath);
  m_connection.configure();
  initialize();
}

// destructor
SiStripCoralIface::~SiStripCoralIface() {	
  LogTrace("SiStripCoralIface") << "[SiStripCoralIface::" << __func__ << "] Destructor called."; 
  // m_connection.close();
}

// open DB connection
void  SiStripCoralIface::initialize() {
  m_session = m_connection.createSession();
  m_session.open(m_connectionString);
  try {
    m_transaction.reset( new cond::DbScopedTransaction(m_session) );
    m_transaction->start(true);
    LogTrace("SiStripCoralIface") << "[SiStripCoralIface::" << __func__ << "] Database connection opened"; 
  }
  catch (...) {
    edm::LogError("SiStripCoralIface") << "Database connection failed";     
    exit(1);
  }
}

// access the status change or lastValue tables
void SiStripCoralIface::doQuery(std::string queryType, coral::TimeStamp startTime, coral::TimeStamp endTime, std::vector<coral::TimeStamp> &vec_changedate, 
				std::vector<float> &vec_actualValue, std::vector<std::string> &vec_dpname)
{
  std::auto_ptr<coral::IQuery> query( m_session.schema(std::string("CMS_TRK_DCS_PVSS_COND")).newQuery());
  std::string condition;

  LogTrace("SiStripCoralIface") << "[SiStripCoralIface::" << __func__ << "] table to be accessed: " << queryType;

  if (queryType == "STATUSCHANGE") {
    query->addToOutputList("FWCAENCHANNEL.CHANGE_DATE","CHANGE_DATE");
    query->addToOutputList("FWCAENCHANNEL.ACTUAL_STATUS","ACTUAL_STATUS");
    query->addToOutputList("DP_NAME2ID.DPNAME","DPNAME");
    query->addToOrderList("FWCAENCHANNEL.CHANGE_DATE");
    query->addToTableList("FWCAENCHANNEL");
    query->addToTableList("DP_NAME2ID");
    condition = "FWCAENCHANNEL.DPID = DP_NAME2ID.id AND FWCAENCHANNEL.CHANGE_DATE<=:tmax AND FWCAENCHANNEL.ACTUAL_STATUS IS NOT NULL AND FWCAENCHANNEL.CHANGE_DATE >=:tmin AND (DP_NAME2ID.dpname like '%easyBoard%')";
  } else if (queryType == "LASTVALUE") {
    query->addToOutputList("DCSLASTVALUE_VOLTAGE.CHANGE_DATE","CHANGE_DATE");
    query->addToOutputList("DCSLASTVALUE_VOLTAGE.ACTUAL_VMON","ACTUAL_VMON");
    query->addToOutputList("DP_NAME2ID.DPNAME","DPNAME");
    query->addToOrderList("DCSLASTVALUE_VOLTAGE.CHANGE_DATE");
    query->addToTableList("DCSLASTVALUE_VOLTAGE");
    query->addToTableList("DP_NAME2ID");
    condition = "DCSLASTVALUE_VOLTAGE.DPID = DP_NAME2ID.id AND DCSLASTVALUE_VOLTAGE.CHANGE_DATE<=:tmax AND DCSLASTVALUE_VOLTAGE.CHANGE_DATE>=:tmin AND DCSLASTVALUE_VOLTAGE.ACTUAL_VMON IS NOT NULL AND (DP_NAME2ID.dpname like '%easyBoard%')";
  }

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
    if( debug_ ) row.toOutputStream( std::cout ) << std::endl;
    numberRow++;
    if (queryType == "STATUSCHANGE") {
      coral::TimeStamp ts =  row["CHANGE_DATE"].data<coral::TimeStamp>();
      vec_changedate.push_back(ts);
      float as = (float)row["ACTUAL_STATUS"].data<float>();
      vec_actualValue.push_back(as);
      std::string id_name = (std::string)row["DPNAME"].data<std::string>();
      vec_dpname.push_back(id_name);
    } else if (queryType == "LASTVALUE") {
      coral::TimeStamp ts =  row["CHANGE_DATE"].data<coral::TimeStamp>();
      vec_changedate.push_back(ts);
      float av = (float)row["ACTUAL_VMON"].data<float>();
      vec_actualValue.push_back(av);
      std::string id_name = (std::string)row["DPNAME"].data<std::string>();
      vec_dpname.push_back(id_name);
    }
  }
  cursor.close();
  LogTrace("SiStripCoralIface") << "[SiStripCoralIface::" << __func__ << "] " << numberRow << " rows retrieved from PVSS Cond DB";
}

// access the channel settings in the status change table
void SiStripCoralIface::doSettingsQuery(coral::TimeStamp startTime, coral::TimeStamp endTime, std::vector<coral::TimeStamp> &vec_changedate,
					std::vector<float> &vec_settings, std::vector<std::string> &vec_dpname, std::vector<uint32_t> &vec_dpid) 
{
  std::auto_ptr<coral::IQuery> query( m_session.schema(std::string("CMS_TRK_DCS_PVSS_COND")).newQuery());
  query->addToOutputList("FWCAENCHANNEL.CHANGE_DATE","CHANGE_DATE");
  query->addToOutputList("FWCAENCHANNEL.SETTINGS_V0","VSET");
  query->addToOutputList("FWCAENCHANNEL.DPID","DPID");
  query->addToOutputList("DP_NAME2ID.DPNAME","DPNAME");
  query->addToOrderList("FWCAENCHANNEL.CHANGE_DATE");
  query->addToTableList("FWCAENCHANNEL");
  query->addToTableList("DP_NAME2ID");
  std::string condition = "FWCAENCHANNEL.DPID = DP_NAME2ID.id AND FWCAENCHANNEL.CHANGE_DATE<=:tmax AND FWCAENCHANNEL.SETTINGS_V0 IS NOT NULL AND FWCAENCHANNEL.CHANGE_DATE >=:tmin AND (DP_NAME2ID.dpname like '%easyBoard%')";

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
    if( debug_ ) row.toOutputStream( std::cout ) << std::endl;
    numberRow++;
    coral::TimeStamp ts =  row["CHANGE_DATE"].data<coral::TimeStamp>();
    vec_changedate.push_back(ts);
    float vs = (float)row["VSET"].data<float>();
    vec_settings.push_back(vs);
    uint32_t id = (uint32_t)row["DPID"].data<float>();
    vec_dpid.push_back(id);
    std::string id_name = (std::string)row["DPNAME"].data<std::string>();
    vec_dpname.push_back(id_name);
  }
  cursor.close();
  LogTrace("SiStripCoralIface") << "[SiStripCoralIface::" << __func__ << "] " << numberRow << " rows retrieved from PVSS Cond DB";
}

void SiStripCoralIface::doNameQuery(std::vector<std::string> &vec_dpname, std::vector<uint32_t> &vec_dpid) 
{
  std::auto_ptr<coral::IQuery> query( m_session.schema(std::string("CMS_TRK_DCS_PVSS_COND")).newQuery());
  query->addToOutputList("DP_NAME2ID.DPNAME","DPNAME");
  query->addToOutputList("DP_NAME2ID.ID","DPID");
  query->addToTableList("DP_NAME2ID");

  std::string condition = "DP_NAME2ID.dpname like '%easyBoard%' ";
  query->setCondition( condition, coral::AttributeList() );

  query->setMemoryCacheSize( 100 );
  coral::ICursor& cursor = query->execute();
  int numberRow=0;
  while( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    numberRow++;
    uint32_t id = (uint32_t)row["DPID"].data<float>();
    vec_dpid.push_back(id);
    std::string id_name = (std::string)row["DPNAME"].data<std::string>();
    vec_dpname.push_back(id_name);
  }
  cursor.close();
  LogTrace("SiStripCoralIface") << "[SiStripCoralIface::" << __func__ << "] " << numberRow << " rows retrieved from PVSS Cond DB";
}
