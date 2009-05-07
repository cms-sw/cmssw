
 /* 
 *  See header file for a description of this class.
 *
 *  $Date: 2009/04/13 20:40:38 $
 *  $Revision: 1.16 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/IMonitoringService.h"
#include "RelationalAccess/IMonitoringReporter.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralBase/MessageStream.h"



#include "CondTools/RPC/interface/RPCIOVReader.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"

#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>
#include <iostream>
#include <sstream>
#include <time.h>
#include "CondFormats/RPCObjects/interface/RPCObFebmap.h"

RPCIOVReader::RPCIOVReader( const std::string& connectionString,
              const std::string& userName,
              const std::string& password):
  RPCDBCom(),
  m_connectionString( connectionString ),
  m_userName( userName ),
  m_password( password )
{}


RPCIOVReader::~RPCIOVReader()
{}

void
RPCIOVReader::run()
{
}

std::vector<unsigned long long>
RPCIOVReader::listIOV()
{
  std::vector<unsigned long long> iov_vect;
  coral::Context::instance().loadComponent("CORAL/Services/ConnectionService");
  coral::Context::instance().loadComponent("COND/Services/SQLMonitoringService");
  coral::Context::instance().loadComponent("COND/Services/XMLAuthenticationService");
  //coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(std::string("/build/gg/key.dat"));
  coral::IHandle<coral::IConnectionService> connectionService=coral::Context::instance().query<coral::IConnectionService>();
  connectionService->configuration().setMonitoringLevel(coral::monitor::Trace);   
  std::string connectionString("sqlite_file:dati.db");
  //std::string connectionString("oracle://cms_orcoff_prep/CMS_COND_PRESH");
  
  coral::ISessionProxy* session = connectionService->connect( connectionString );
  session->transaction().start();
  coral::ISchema& schema = session->nominalSchema();
  coral::IQuery* query = schema.newQuery();
  query->addToTableList( "IOV_DATA" );
  query->addToOutputList( "IOV_DATA.IOV_TIME", "IOV" );
  std::string cond = "IOV_DATA.IOV_TIME is not NULL";
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    unsigned long long iov = row["IOV"].data<unsigned long long>();
    //std::cout << ">> IOV: " << iov << std::endl;
    iov_vect.push_back(iov);
  }
  delete query;
  session->transaction().commit();
  delete session;
  
  return iov_vect;

}




std::vector<RPCObImon::I_Item>
RPCIOVReader::getIMON(unsigned long long IMIN, unsigned long long IMAX)
{
  IMIN = IMIN - 1;
  IMAX = IMAX + 1;
  coral::Context::instance().loadComponent("CORAL/Services/ConnectionService");
  coral::Context::instance().loadComponent("COND/Services/SQLMonitoringService");
  coral::Context::instance().loadComponent("COND/Services/XMLAuthenticationService");
  //coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(std::string("/build/gg/key.dat"));
  coral::IHandle<coral::IConnectionService> connectionService=coral::Context::instance().query<coral::IConnectionService>();
  connectionService->configuration().setMonitoringLevel(coral::monitor::Trace);   
  std::string connectionString("sqlite_file:dati.db");
  //std::string connectionString("oracle://cms_orcoff_prep/CMS_COND_PRESH");
  
  coral::ISessionProxy* session = connectionService->connect( connectionString );
  session->transaction().start();
  coral::ISchema& schema = session->nominalSchema();
  coral::IQuery* query = schema.newQuery();
  
  coral::AttributeList conditionData;
  conditionData.extend<unsigned long long>( "Imin" );
  conditionData.extend<unsigned long long>( "Imax" );
  conditionData["Imin"].data<unsigned long long>() = IMIN;
  conditionData["Imax"].data<unsigned long long>() = IMAX;
  
  query->addToTableList( "IOV_DATA" );
  query->addToOutputList( "IOV_DATA.IOV_TIME", "IOV" );
  query->addToTableList( "RPCOBIM_OBIMON_RPC" );
  query->addToOutputList( "RPCOBIM_OBIMON_RPC.DOBIMON_RPC_CV_DPID", "DPID" );
  query->addToOutputList( "RPCOBIM_OBIMON_RPC.DOBIMON_RPC_CV_DAY", "DAY" );
  query->addToOutputList( "RPCOBIM_OBIMON_RPC.DOBIMON_RPC_CV_TIME", "TIME" );
  query->addToOutputList( "RPCOBIM_OBIMON_RPC.DOBIMON_RPC_CV_VALUE", "IMON" );
  std::string condition = "RPCOBIM_OBIMON_RPC.DOBIMON_RPC_CV_VALUE IS NOT NULL AND IOV >:Imin AND IOV <:Imax";
  query->setCondition( condition , conditionData );
  coral::ICursor& cursor = query->execute();
  int nrow = 1;
  RPCObImon::I_Item Idata;
  std::vector<RPCObImon::I_Item> Imon_v;
  while ( cursor.next() ) {
    nrow++;
    const coral::AttributeList& row = cursor.currentRow();
    int   dpid  = row["DPID"].data<int>();
    float value = row["IMON"].data<float>();
    int   day   = row["DAY"].data<int>();
    int   time  = row["TIME"].data<int>();
    //unsigned long long wiov = row["IOV"].data<unsigned long long>();
    //std::cout << nrow << " " << value << " " << wiov << std::endl;
    Idata.dpid  = dpid;
    Idata.value = value;
    Idata.day   = day;
    Idata.time  = time;
    Imon_v.push_back(Idata);
  }
  delete query;
  session->transaction().commit();
  delete session;

  return Imon_v;
}



std::string 
RPCIOVReader::toDay(int intday) {
 
  std::ostringstream day_;
  day_<< intday;
  std::string day = day_.str();

  while (day.length() != 6) {
    day.insert(0,"0");
  }
  
  day.insert(2,"/");
  day.insert(5,"/");

  return day;
}



std::string 
RPCIOVReader::toTime(int inttime) {
 
  std::ostringstream time_;
  time_<< inttime;
  std::string time = time_.str();

  while (time.length() != 6) {
    time.insert(0,"0");
  }
  
  time.insert(2,":");
  time.insert(5,".");

  return time;
}
